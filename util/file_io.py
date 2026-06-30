import threading
import glob
import pickle
import os
import numpy as np
import h5py

class ChunkPrefetcher:
    """Prefetches the next buffer chunk file in a background thread."""

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self._thread = None
        self._result = {}

    def _load(self, sub_files):
        self._result = {}
        if len(sub_files) == 0:
            self._result["stats"] = {"actions": []}
            self._result["path"] = []
            return

        all_stats = []
        for path in sub_files:
            with open(path, "rb") as f:
                all_stats.append(pickle.load(f))

        merged = {}
        merged["states"] = {
            k: np.concatenate([s["states"][k] for s in all_stats], axis=0)
            for k in all_stats[0]["states"]
        }
        for key in ["actions", "rewards", "done"]:
            merged[key] = np.concatenate([s[key] for s in all_stats], axis=0)

        self._result["stats"] = merged
        self._result["path"] = sub_files  # list of loaded paths for logging

    def prefetch(self, load_chunk_num=4):
        """Randomly sample load_chunk_num sub-chunk files from all available ones.

        Args:
            load_chunk_num: number of pickle files to load and merge
        """
        all_files = glob.glob(os.path.join(self.checkpoint_dir, "buffer_chunk*_*.pkl"))
        if len(all_files) == 0:
            selected = []
        elif load_chunk_num >= len(all_files):
            selected = all_files
        else:
            selected = np.random.choice(all_files, size=load_chunk_num, replace=False).tolist()
        self._thread = threading.Thread(target=self._load, args=(selected,))
        self._thread.start()

    def get(self):
        """Block until the prefetched result is ready and return (stats, path).

        Args:
            prefetch_next: if True, automatically start loading index+1 in the
                           background before returning.
        """
        if self._thread is not None:
            self._thread.join()
        stats, path = self._result["stats"], self._result.get("path")
        return stats, path
    
    def save_chunk(self, chunk_idx, num_traj_per_chunk, stats):
        num_trajs = stats["done"].shape[0]
        total_chunks = num_trajs // num_traj_per_chunk
        obs_keys = stats['states'].keys()
        for k in range(total_chunks):
            traj_start, traj_end = k * num_traj_per_chunk, (k+1) * num_traj_per_chunk
            
            chunk_file_name = os.path.join(self.checkpoint_dir, f"buffer_chunk{chunk_idx}_{k}.pkl" )
            if os.path.exists(chunk_file_name):
                os.remove(chunk_file_name)
            
            fresh_stats = {}
            # merge: handle states dict separately
            fresh_stats["states"] = {
                ob_k: stats["states"][ob_k][traj_start:traj_end]
                for ob_k in obs_keys
            }
            for key in ["actions", "rewards", "done"]:
                fresh_stats[key] = stats[key][traj_start:traj_end]
            
            with open(chunk_file_name, "wb") as f:
                pickle.dump(fresh_stats, f, protocol=pickle.HIGHEST_PROTOCOL)


class HDF5Fetcher:
    """Single ring-buffer replay queue backed by an HDF5 file on disk.

    All trajectories are stored in buffer.h5. The ring buffer has capacity
    max_buffer_num: a write_index advances modulo max_buffer_num so new data
    overwrites the oldest entries once full. Both write_index and traj_count
    are persisted as HDF5 attributes so the queue survives process restarts.
    """

    def __init__(self, checkpoint_dir, max_buffer_num):
        self.checkpoint_dir = checkpoint_dir
        self.max_buffer_num = max_buffer_num
        self._thread = None
        self._result = {}
        self._write_index = 0
        self._traj_count = 0
        self._lock = threading.Lock()
        self._load_ring_state()

    @property
    def total_trajs(self):
        return self._traj_count

    def _path(self):
        return os.path.join(self.checkpoint_dir, "buffer.h5")

    # ------------------------------------------------------------------
    # Ring-buffer state helpers
    # ------------------------------------------------------------------

    def _load_ring_state(self):
        path = self._path()
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                self._write_index = int(f.attrs.get("write_index", 0))
                self._traj_count = int(f.attrs.get("traj_count", 0))

    def _flush_ring_state(self, f):
        f.attrs["write_index"] = self._write_index
        f.attrs["traj_count"] = self._traj_count

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------

    def _write_ring(self, stats):
        """Write stats into the ring buffer."""
        n = stats["done"].shape[0]
        if n == 0:
            return
        path = self._path()
        with self._lock:
            if not os.path.exists(path):
                with h5py.File(path, "w") as f:
                    grp = f.create_group("states")
                    for ob_k, data in stats["states"].items():
                        grp.create_dataset(
                            ob_k,
                            shape=(self.max_buffer_num,) + data.shape[1:],
                            dtype=data.dtype,
                        )
                    for field in ["actions", "rewards", "done"]:
                        data = stats[field]
                        f.create_dataset(
                            field,
                            shape=(self.max_buffer_num,) + data.shape[1:],
                            dtype=data.dtype,
                        )
                    self._flush_ring_state(f)
            with h5py.File(path, "a") as f:
                i = 0
                while i < n:
                    space = self.max_buffer_num - self._write_index
                    chunk = min(n - i, space)
                    start, end = self._write_index, self._write_index + chunk
                    for ob_k in stats["states"]:
                        f["states"][ob_k][start:end] = stats["states"][ob_k][i:i + chunk]
                    for field in ["actions", "rewards", "done"]:
                        f[field][start:end] = stats[field][i:i + chunk]
                    self._write_index = (self._write_index + chunk) % self.max_buffer_num
                    self._traj_count = min(self._traj_count + chunk, self.max_buffer_num)
                    i += chunk
                self._flush_ring_state(f)

    def _read_ring(self, n_sample):
        """Return n_sample randomly sampled trajectories from the ring buffer, or None if empty."""
        with self._lock:
            n = self._traj_count
            path = self._path()
            if not os.path.exists(path) or n == 0:
                return None
            if n_sample <= n:
                # HDF5 fancy indexing requires strictly increasing (no duplicate) indices.
                idx = np.random.choice(n, size=n_sample, replace=False)
                idx.sort()
                with h5py.File(path, "r") as f:
                    stat = {"states": {ob_k: f["states"][ob_k][idx] for ob_k in f["states"]}}
                    for field in ["actions", "rewards", "done"]:
                        stat[field] = f[field][idx]
            else:
                # Need more samples than available — read all rows then resample in numpy.
                with h5py.File(path, "r") as f:
                    stat = {"states": {ob_k: f["states"][ob_k][:n] for ob_k in f["states"]}}
                    for field in ["actions", "rewards", "done"]:
                        stat[field] = f[field][:n]
                idx = np.random.choice(n, size=n_sample, replace=True)
                stat = {
                    "states": {ob_k: stat["states"][ob_k][idx] for ob_k in stat["states"]},
                    **{field: stat[field][idx] for field in ["actions", "rewards", "done"]},
                }
            return stat

    def _load(self, num_trajs):
        self._result = {}
        half = num_trajs // 2

        with self._lock:
            n = self._traj_count
            path = self._path()
            if not os.path.exists(path) or n == 0:
                self._result["stats"] = {"actions": []}
                return

            # Pass 1: load only rewards to decide which indices to fetch.
            with h5py.File(path, "r") as f:
                rewards = f["rewards"][:n]

            traj_rewards = rewards.reshape(n, -1).sum(axis=1)
            success_idx = np.where(traj_rewards != 0)[0]
            failure_idx = np.where(traj_rewards == 0)[0]

            def _choose(indices, k):
                if len(indices) == 0:
                    return None
                chosen = np.random.choice(indices, size=k, replace=len(indices) < k)
                chosen.sort()
                return chosen

            groups = [g for g in (_choose(success_idx, half), _choose(failure_idx, half))
                      if g is not None]
            if not groups:
                self._result["stats"] = {"actions": []}
                return
            idx = np.concatenate(groups)
            idx.sort()
            unique_idx = np.unique(idx)
            remap = np.searchsorted(unique_idx, idx)

            # Pass 2: load full data only for selected indices.
            with h5py.File(path, "r") as f:
                stat = {"states": {ob_k: f["states"][ob_k][unique_idx][remap] for ob_k in f["states"]}}
                for field in ["actions", "rewards", "done"]:
                    stat[field] = f[field][unique_idx][remap]

        self._result["stats"] = stat

    def prefetch(self, num_trajs):
        """Sample num_trajs trajectories from the ring buffer in a background thread."""
        self._thread = threading.Thread(target=self._load, args=(num_trajs,))
        self._thread.start()

    def get(self):
        """Block until the prefetched result is ready and return stats."""
        if self._thread is not None:
            self._thread.join()
        return self._result["stats"]

    def save_chunk(self, stats):
        """Write all trajectories into the ring buffer."""
        self._write_ring(stats)
