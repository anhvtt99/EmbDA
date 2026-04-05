import numpy as np
import faiss

def _to_float32_contiguous(x):
    return np.ascontiguousarray(x.astype(np.float32))

def faiss_L2_search(xq, xb, k):
    """
    xb: database
    xq: queries
    k: k nearest neighbors
    """
    xb = _to_float32_contiguous(xb)
    xq = _to_float32_contiguous(xq)

    index = faiss.IndexFlatL2(xb.shape[1])

    if hasattr(faiss, "StandardGpuResources"):
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(xb)
    D, I = index.search(xq, k)
    return I

def faiss_cos_search(xq, xb, k):
    """
    xb: database
    xq: queries
    k: k nearest neighbors
    """
    xb = _to_float32_contiguous(xb)
    xq = _to_float32_contiguous(xq)

    index = faiss.IndexFlatIP(xb.shape[1])

    if hasattr(faiss, "StandardGpuResources"):
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(xb)
    D, I = index.search(xq, k)
    return I