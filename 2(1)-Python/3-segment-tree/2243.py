from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    # 구현하세요!
    box: SegmentTree = SegmentTree(
        data=[0] * 1000000,  # 1부터 1,000,000까지의 캔디 맛 개수 초기화
        default=lambda: 0,   # 기본값 0
        merge=lambda x, y: x + y  # 병합 함수: 두 값을 더함
    )

    n = int(sys.stdin.readline().strip())  # 명령 수 입력
    answers = []

    for _ in range(n):
        line = list(map(int, sys.stdin.readline().strip().split()))

        if line[0] == 2:  # 캔디 추가 명령
            which_taste = line[1]
            how_many = line[2]
            box.update(which_taste, box.query(which_taste, which_taste, 1, 1, 1000000) + how_many, 1, 1, 1000000)
        else:  # 캔디 뽑기 명령
            rank = line[1]
            left, right = 1, 1000000
            answer = -1

            while left <= right:
                mid = (left + right) // 2
                if box.query(1, mid, 1, 1, 1000000) >= rank:
                    answer = mid
                    right = mid - 1
                else:
                    left = mid + 1

            box.update(answer, box.query(answer, answer, 1, 1, 1000000) - 1, 1, 1, 1000000)
            answers.append(answer)

    for ans in answers:
        print(ans)


if __name__ == "__main__":
    main()