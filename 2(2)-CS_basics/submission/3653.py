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


import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    """
    입력 형식:
    - 첫 번째 줄: 테스트 케이스 개수
    - 각 테스트 케이스는 두 줄로 구성:
        1) n m (영화의 개수 n, 요청의 개수 m)
        2) 요청된 영화 번호 m개

    처리 과정:
    1. 각 테스트 케이스에 대해 초기 배열, 위치 배열 설정.
    2. 세그먼트 트리 생성, 각 요청에 대해 위에 쌓여 있는 영화의 개수를 계산.
    3. 쌓여 있는 상태 변경점 반영.

    출력:
    - 각 테스트 케이스의 결과를 한 줄로 출력.
    """
    iter = int(sys.stdin.readline().strip())  # 테스트 케이스 개수
    lines = []
    for _ in range(iter):
        n, m = map(int, sys.stdin.readline().strip().split())
        queries = list(map(int, sys.stdin.readline().strip().split()))
        lines.append((n, m, queries))

    results = []

    for n, m, queries in lines:
        arr = [0] * (n + m + 1)  # 세그먼트 트리에 쓰일 배열
        pos = [0] * (n + 1)  # 영화의 현재 위치를 저장하는 배열

        # 초기 배열 설정
        for i in range(1, n + 1):
            arr[m + i] = 1
            pos[i] = m + i

        # 세그먼트 트리 생성
        t:SegmentTree = SegmentTree(
            data=arr,
            default=lambda: 0,  # 기본값 0
            merge=lambda x, y: x + y  # 병합 함수: 합산
        )

        answers = []
        current_top = m

        for movie in queries:
            current_pos = pos[movie]
            # 현재 영화 위에 있는 영화 개수 쿼리
            answers.append(t.query(1, current_pos - 1, 1, 1, len(arr) - 1))
            
            # 현재 위치에서 제거
            t.update(current_pos, 0, 1, 1, len(arr) - 1)
            # 맨 위에 추가
            t.update(current_top, 1, 1, 1, len(arr) - 1)
            # 영화 위치 갱신
            pos[movie] = current_top
            current_top -= 1

        results.append(answers)

    for result in results:
        print(" ".join(map(str, result)))


if __name__ == "__main__":
    main()