from itertools import zip_longest
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import jittor as jt
import math


class Pointclouds:
    def __init__(self, points, normals=None, features=None) -> None:  # syh: feature 有梯度
        self.equisized = False
        self.valid = None

        self._N = 0  # batch size (number of clouds)
        self._P = 0  # (max) number of points per cloud
        self._C = None  # number of channels in the features

        # List of Tensors of points and features.
        self._points_list = None
        self._normals_list = None
        self._features_list = None

        # Number of points per cloud.
        self._num_points_per_cloud = None  # N

        # Packed representation.
        self._points_packed = None  # (sum(P_n), 3)
        self._normals_packed = None  # (sum(P_n), 3)
        self._features_packed = None  # (sum(P_n), C)

        self._packed_to_cloud_idx = None  # sum(P_n)

        # Index of each cloud's first point in the packed points.
        # Assumes packing is sequential.
        self._cloud_to_packed_first_idx = None  # N

        # Padded representation.
        self._points_padded = None  # (N, max(P_n), 3)
        self._normals_padded = None  # (N, max(P_n), 3)
        self._features_padded = None  # (N, max(P_n), C)

        # Index to convert points from flattened padded to packed.
        self._padded_to_packed_idx = None  # N * max_P

        # Identify type of points.
        if isinstance(points, list):
            self._points_list = points
            self._N = len(self._points_list)
            self.valid = jt.zeros((self._N,), dtype='bool')

            if self._N > 0:
                # self.device = self._points_list[0].device
                for p in self._points_list:
                    if len(p) > 0 and (p.dim() != 2 or p.shape[1] != 3):
                        raise ValueError("Clouds in list must be of shape Px3 or empty")

                num_points_per_cloud = jt.array(
                    [len(p) for p in self._points_list]
                )
                
                self._P = int(num_points_per_cloud.max())
                self.valid = jt.array(
                    [len(p) > 0 for p in self._points_list],
                    dtype='bool',
                )

                if len(num_points_per_cloud.unique()) == 1:
                    self.equisized = True
                self._num_points_per_cloud = num_points_per_cloud
            else:
                self._num_points_per_cloud = jt.array([], dtype='int64')

        elif jt.is_var(points):
            if points.dim() != 3 or points.shape[2] != 3:
                raise ValueError("Points tensor has incorrect dimensions.")
            self._points_padded = points
            self._N = self._points_padded.shape[0]
            self._P = self._points_padded.shape[1]
            self.valid = jt.ones((self._N,), dtype='bool')
            self._num_points_per_cloud = jt.array(
                [self._P] * self._N
            )
            self.equisized = True
        else:
            raise ValueError(
                "Points must be either a list or a tensor with \
                    shape (batch_size, P, 3) where P is the maximum number of \
                    points in a cloud."
            )

        # parse normals
        normals_parsed = self.parse_auxiliary_input(normals)
        self._normals_list, self._normals_padded, normals_C = normals_parsed
        if normals_C is not None and normals_C != 3:
            raise ValueError("Normals are expected to be 3-dimensional")

        # parse features
        features_parsed = self.parse_auxiliary_input(features)  # syh: 这个有梯度，所以需要从这里面拿,好，现在假设features_parsed有梯度
        self.r_features_list, self._features_padded, features_C = features_parsed
        if features_C is not None:
            self._C = features_C

    def parse_auxiliary_input(
        self, aux_input
    ) -> Tuple[Optional[List[jt.Var]], Optional[jt.Var], Optional[int]]:
        if aux_input is None or self._N == 0:
            return None, None, None

        aux_input_C = None

        if isinstance(aux_input, list):  # syh: 正常传入的feature应该是列表的形式，那么这里是正常的
            return self.parse_auxiliary_input_list(aux_input)
        if jt.Var(aux_input):
            if aux_input.dim() != 3:
                raise ValueError("Auxiliary input tensor has incorrect dimensions.")
            if self._N != aux_input.shape[0]:
                raise ValueError("Points and inputs must be the same length.")
            if self._P != aux_input.shape[1]:
                raise ValueError(
                    "Inputs tensor must have the right maximum \
                    number of points in each cloud."
                )
            aux_input_C = aux_input.shape[2]
            return None, aux_input, aux_input_C
        else:
            raise ValueError(
                "Auxiliary input must be either a list or a tensor with \
                    shape (batch_size, P, C) where P is the maximum number of \
                    points in a cloud."
            )

    def parse_auxiliary_input_list(
            self, aux_input: list
    ) -> Tuple[Optional[List[jt.Var]], None, Optional[int]]:
        """
        Interpret the auxiliary inputs (normals, features) given to __init__,
        if a list.

        Args:
            aux_input:
                - List where each element is a tensor of shape (num_points, C)
                  containing the features for the points in the cloud.
              For normals, C = 3

        Returns:
            3-element tuple of list, padded=None, num_channels.
            If aux_input is list, then padded is None. If aux_input is a tensor,
            then list is None.
        """
        aux_input_C = None
        good_empty = None
        needs_fixing = False

        if len(aux_input) != self._N:
            raise ValueError("Points and auxiliary input must be the same length.")
        for p, d in zip(self._num_points_per_cloud, aux_input):  # syh:aux_input是有梯度的
            valid_but_empty = p == 0 and d is not None and d.ndim == 2
            if p > 0 or valid_but_empty:
                if p != d.shape[0]:
                    raise ValueError(
                        "A cloud has mismatched numbers of points and inputs"
                    )
                if d.dim() != 2:
                    raise ValueError(
                        "A cloud auxiliary input must be of shape PxC or empty"
                    )
                if aux_input_C is None:
                    aux_input_C = d.shape[1]
                elif aux_input_C != d.shape[1]:
                    raise ValueError("The clouds must have the same number of channels")
            else:
                needs_fixing = True

        if aux_input_C is None:
            # We found nothing useful
            return None, None, None

        # If we have empty but "wrong" inputs we want to store "fixed" versions.
        if needs_fixing:
            if good_empty is None:
                good_empty = jt.array((0, aux_input_C))
            aux_input_out = []
            for p, d in zip(self._num_points_per_cloud, aux_input):  # syh: 看aux，怎么感觉这个是正常的？
                valid_but_empty = p == 0 and d is not None and d.ndim == 2
                if p > 0 or valid_but_empty:
                    aux_input_out.append(d)
                else:
                    aux_input_out.append(good_empty)
        else:
            aux_input_out = aux_input

        return aux_input_out, None, aux_input_C  # syh:关键是第一个，一定有梯度才行

    def __len__(self) -> int:
        return self._N

    # def __getitem__(  # syh:可能不需要
    #         self,
    #         index: Union[int, List[int], slice, jt.Var, jt.Var],  # torch.BoolTensor, torch.LongTensor
    # ) -> "Pointclouds":
    #     """
    #     Args:
    #         index: Specifying the index of the cloud to retrieve.
    #             Can be an int, slice, list of ints or a boolean tensor.
    #
    #     Returns:
    #         Pointclouds object with selected clouds. The tensors are not cloned.
    #     """
    #     normals, features = None, None
    #     normals_list = self.normals_list()  # syh: 这里改一下
    #     features_list = self.features_list()
    #     if isinstance(index, int):
    #         points = [self.points_list()[index]]
    #         if normals_list is not None:
    #             normals = [normals_list[index]]
    #         if features_list is not None:
    #             features = [features_list[index]]
    #     elif isinstance(index, slice):
    #         points = self.points_list()[index]
    #         if normals_list is not None:
    #             normals = normals_list[index]
    #         if features_list is not None:
    #             features = features_list[index]
    #     elif isinstance(index, list):
    #         points = [self.points_list()[i] for i in index]
    #         if normals_list is not None:
    #             normals = [normals_list[i] for i in index]
    #         if features_list is not None:
    #             features = [features_list[i] for i in index]
    #     elif isinstance(index, torch.Tensor):
    #         if index.dim() != 1 or index.dtype.is_floating_point:
    #             raise IndexError(index)
    #         # NOTE consider converting index to cpu for efficiency
    #         if index.dtype == torch.bool:
    #             # advanced indexing on a single dimension
    #             index = index.nonzero()
    #             index = index.squeeze(1) if index.numel() > 0 else index
    #             index = index.tolist()
    #         points = [self.points_list()[i] for i in index]
    #         if normals_list is not None:
    #             normals = [normals_list[i] for i in index]
    #         if features_list is not None:
    #             features = [features_list[i] for i in index]
    #     else:
    #         raise IndexError(index)
    #
    #     return self.__class__(points=points, normals=normals, features=features)

    def normals_list(self) -> Optional[List[jt.Var]]:
        """
                Get the list representation of the normals,
                or None if there are no normals.

                Returns:
                    list of tensors of normals of shape (P_n, 3).
                """
        if self._normals_list is None:
            if self._normals_padded is None:
                # No normals provided so return None
                return None
            self._normals_list = self.padded_to_list(  # syh: strcut_utils补一下
                self._normals_padded, self.num_points_per_cloud().tolist()
            )
        return self._normals_list

    def features_list(self) -> Optional[List[jt.Var]]:
            """
            Get the list representation of the features,
            or None if there are no features.

            Returns:
                list of tensors of features of shape (P_n, C).
            """
            if self.r_features_list is None:
                if self._features_padded is None:
                    # No features provided so return None
                    return None
                self.r_features_list = self.padded_to_list(
                    self._features_padded, self.num_points_per_cloud().tolist()
                )
            return self.r_features_list

    def features_packed(self) -> Optional[jt.Var]:
        self._compute_packed()
        return self._features_packed

    def points_packed(self) -> jt.Var:  # syh: 这个要用
        self._compute_packed()
        return self._points_packed

    def cloud_to_packed_first_idx(self):  # syh: 这个要用
        """
        Return a 1D tensor x with length equal to the number of clouds such that
        the first point of the ith cloud is points_packed[x[i]].

        Returns:
            1D tensor of indices of first items.
        """
        self._compute_packed()
        return self._cloud_to_packed_first_idx

    def _compute_packed(self, refresh: bool = False):  # syh:这个改了说不定就没事了
        """
        Computes the packed version from points_list, normals_list and
        features_list and sets the values of auxiliary tensors.

        Args:
            refresh: Set to True to force recomputation of packed
                representations. Default: False.
        """

        if not (
                refresh
                or any(
            v is None
            for v in [
                self._points_packed,
                self._packed_to_cloud_idx,
                self._cloud_to_packed_first_idx,
            ]
        )
        ):
            return

        # Packed can be calculated from padded or list, so can call the
        # accessor function for the lists.
        points_list = self.points_list()
        normals_list = self.normals_list()
        features_list = self.features_list()
        if self.isempty():
            self._points_packed = jt.zeros(
                (0, 3), dtype='float32'
            )
            self._packed_to_cloud_idx = jt.zeros(
                (0,), dtype='int64'
            )
            self._cloud_to_packed_first_idx = jt.zeros(
                (0,), dtype='int64'
            )
            self._normals_packed = None
            self._features_packed = None
            return

        points_list_to_packed = self.list_to_packed(points_list)
        self._points_packed = points_list_to_packed[0]
        # if not torch.allclose(self._num_points_per_cloud, points_list_to_packed[1]):  # sth: 这个找一下对应啊
        #     raise ValueError("Inconsistent list to packed conversion")
        self._cloud_to_packed_first_idx = points_list_to_packed[2]
        self._packed_to_cloud_idx = points_list_to_packed[3]

        self._normals_packed, self._features_packed = None, None
        if normals_list is not None:
            normals_list_to_packed = self.list_to_packed(normals_list)
            self._normals_packed = normals_list_to_packed[0]

        if features_list is not None:
            features_list_to_packed = self.list_to_packed(features_list)
            self._features_packed = features_list_to_packed[0]

    def points_list(self) -> List[jt.Var]:

        if self._points_list is None:
            assert (
                    self._points_padded is not None
            ), "points_padded is required to compute points_list."
            points_list = []
            for i in range(self._N):
                points_list.append(
                    self._points_padded[i, : self.num_points_per_cloud()[i]]
                )
            self._points_list = points_list
        return self._points_list

    def num_points_per_cloud(self) -> jt.Var:
        """
        Return a 1D tensor x with length equal to the number of clouds giving
        the number of points in each cloud.

        Returns:
            1D tensor of sizes.
        """
        return self._num_points_per_cloud

    def padded_to_list(self, x: jt.Var, split_size: Union[Sequence[int], Sequence[Sequence[int]], None] = None,) -> jt.Var:
        x_list = list(x.unbind(0))
        if split_size is None:
            return x_list
        N = len(split_size)
        if x.shape[0] != N:
            raise ValueError("Split size must be of same length as inputs first dimension")

        for i in range(N):
                if isinstance(split_size[i], int):
                    x_list[i] = x_list[i][: split_size[i]]
                else:
                    slices = tuple(slice(0, s) for s in split_size[i])  # pyre-ignore
                    x_list[i] = x_list[i][slices]
        return x_list

    def list_to_packed(self, x: List[jt.Var]):
        if not x:
            raise ValueError("Input list is empty")
        sizes = [xi.shape[0] for xi in x]
        sizes_total = sum(sizes)
        num_items = jt.array(sizes, dtype='int64')
        item_packed_first_idx = jt.zeros_like(num_items)
        item_packed_first_idx[1:] = jt.cumsum(num_items[:-1], dim=0)
        item_packed_to_list_idx = jt.arange(
            sizes_total, dtype=jt.int64
        )
        item_packed_to_list_idx = (  # syh:这里可能改起来有问题
                self.bucketize(item_packed_to_list_idx, item_packed_first_idx, right=True) - 1
        )
        x_packed = jt.cat(x, dim=0)

        return x_packed, num_items, item_packed_first_idx, item_packed_to_list_idx

    def isempty(self) -> bool:
        return self._N == 0 or self.valid.equal(False).all()

    def bucketize(self, a: jt.Var, boundaries, *, out_int32: bool = False, right: bool = False,):
        if boundaries.dim() != 1:
            print(f"boundaries tensor must be 1 dimension but got dim({boundaries.dim()})")

        out_dtype = jt.int32 if out_int32 else jt.int64
        n_boundaries = boundaries.shape[-1]
        if n_boundaries == 0:
            return jt.zeros_like(a)
        start = jt.zeros(a.shape, dtype='int64')
        end = start + n_boundaries
        mid = start + (end - start) // 2
        mid_val = boundaries[mid]
        if right:
            cond_mid = mid_val > a
        else:
            cond_mid = mid_val >= a
        start = jt.where(cond_mid, start, mid + 1)
        if n_boundaries > 1:
            cond_update = jt.ones_like(a, 'bool')
            niters = int(math.log2(n_boundaries))
            for _ in range(niters):
                end = jt.where(cond_mid & cond_update, mid, end)
                cond_update = start < end
                # start might end up pointing to 1 past the end, we guard against that
                mid = jt.where(cond_update, start + (end - start) // 2, 0)
                mid_val = boundaries[mid]
                # If right is true, the buckets are closed on the *left*
                # (i.e., we are doing the equivalent of std::upper_bound in C++)
                # Otherwise they are closed on the right (std::lower_bound)
                if right:
                    cond_mid = mid_val > a
                else:
                    cond_mid = mid_val >= a
                start = jt.where((~cond_mid) & cond_update, mid + 1, start)

        return start.to(dtype=out_dtype)

    def padded_to_packed_idx(self):
        """
        Return a 1D tensor x with length equal to the total number of points
        such that points_packed()[i] is element x[i] of the flattened padded
        representation.
        The packed representation can be calculated as follows.

        .. code-block:: python

            p = points_padded().reshape(-1, 3)
            points_packed = p[x]

        Returns:
            1D tensor of indices.
        """
        if self._padded_to_packed_idx is not None:
            return self._padded_to_packed_idx
        if self._N == 0:
            self._padded_to_packed_idx = []
        else:
            self._padded_to_packed_idx = jt.cat(
                [
                    jt.arange(v) + i * self._P
                    for (i, v) in enumerate(self.num_points_per_cloud())
                ],
                dim=0,
            )
        return self._padded_to_packed_idx
