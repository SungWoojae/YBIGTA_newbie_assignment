from lib import SegmentTree
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