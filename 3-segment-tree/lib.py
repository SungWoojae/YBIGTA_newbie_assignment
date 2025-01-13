from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable, List

"""
TODO:
- SegmentTree 구현하기
"""


T = TypeVar("T")
U = TypeVar("U")




class SegmentTree(Generic[T, U]):
    # 구현하세요!
    def __init__(
        self, 
        data: List[T], 
        default: Callable[[], T], 
        merge: Callable[[T, T], T]
    ):
        """
        세그먼트 트리 초기화

        Args:
            data (List[T]): 초기 입력 데이터
            default (Callable[[], T]): 기본값 반환 함수 (범위 밖 노드의 값)
            merge (Callable[[T, T], T]): 두 노드를 병합하는 함수
        """
        self.n = len(data)
        self.default = default
        self.merge = merge
        self.tree = [self.default() for _ in range(4 * self.n)]
        self._build(data, 1, 1, self.n)

    def _build(self, data: List[T], node: int, start: int, end: int):
        """
        세그먼트 트리 생성
        """
        if start == end:
            # 리프 노드
            self.tree[node] = data[start - 1]
        else:
            mid = (start + end) // 2
            left_child = 2 * node
            right_child = 2 * node + 1

            self._build(data, left_child, start, mid)
            self._build(data, right_child, mid + 1, end)

            self.tree[node] = self.merge(self.tree[left_child], self.tree[right_child])

    def update(self, index: int, value: T, node: int, start: int, end: int):
        """
        노드 업데이트
        """
        if start == end:
            self.tree[node] = value
        else:
            mid = (start + end) // 2
            left_child = 2 * node
            right_child = 2 * node + 1

            if start <= index <= mid:
                self.update(index, value, left_child, start, mid)
            else:
                self.update(index, value, right_child, mid + 1, end)

            self.tree[node] = self.merge(self.tree[left_child], self.tree[right_child])

    def query(self, left: int, right: int, node: int, start: int, end: int) -> T:
        """
        범위 쿼리
        """
        if right < start or left > end:
            return self.default()

        if left <= start and end <= right:
            return self.tree[node]

        mid = (start + end) // 2
        left_child = 2 * node
        right_child = 2 * node + 1

        left_result = self.query(left, right, left_child, start, mid)
        right_result = self.query(left, right, right_child, mid + 1, end)

        return self.merge(left_result, right_result)