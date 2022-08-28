
import torch
import semi_global_matching_cuda

def mkDispmap(left, right, pathA, return_gpu=True):
    assert len(left.shape) == 2
    assert len(right.shape) == 2
    assert (left.shape[0] % 4 == 0) and (left.shape[1] % 4 == 0)
    assert (right.shape[0] % 4 == 0) and (right.shape[1] % 4 == 0)

    H, W = left.shape

    left = left.cuda()
    right = right.cuda()


    if pathA == 2:
        dispmap = semi_global_matching_cuda.sgm(left, right, 7, 84, H, W, 2)
    elif pathA == 4:
        dispmap = semi_global_matching_cuda.sgm(left, right, 7, 92, H, W, 4)
    elif pathA == 8:
        dispmap = semi_global_matching_cuda.sgm(left, right, 6, 96, H, W, 8)

    if return_gpu:
        return dispmap
    else:
        return dispmap.cpu()

if __name__ == '__main__':
    print('hi')
