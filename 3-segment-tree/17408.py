from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


class Pair(tuple[int, int]):
    """
    힌트: 2243, 3653에서 int에 대한 세그먼트 트리를 만들었다면 여기서는 Pair에 대한 세그먼트 트리를 만들 수 있을지도...?
    """
    def __new__(cls, a: int, b: int) -> 'Pair':
        return super().__new__(cls, (a, b))

    @staticmethod
    def default() -> 'Pair':
        """
        기본값
        이게 왜 필요할까...?
        ㄴ> 쿼리 시 노드의 범위에서 벗어난 경우 반환할 값
        """
        return Pair(0, 0)

    @staticmethod
    def f_conv(w: int) -> 'Pair':
        """
        원본 수열의 값을 대응되는 Pair 값으로 변환하는 연산
        이게 왜 필요할까...?
        ㄴ> 리프 노드를 표현하기 위해
        """
        return Pair(w, 0)

    @staticmethod
    def f_merge(a: 'Pair', b: 'Pair') -> 'Pair':
        """
        두 Pair를 하나의 Pair로 합치는 연산
        이게 왜 필요할까...?
        ㄴ> 부모 노드의 값을 구하기 위해
        """
        return Pair(*sorted([*a, *b], reverse=True)[:2])

    def sum(self) -> int:
        return self[0] + self[1]


def main() -> None:
    # 구현하세요!
    n = int(input())
    data = [Pair.f_conv(int(x)) for x in input().split()]
    results=[]

    t:SegmentTree = SegmentTree(data,Pair.default,Pair.f_merge)

    q = int(input())
    for _ in range(q):
        query = [int(x) for x in sys.stdin.readline().strip().split(' ')]
        if query[0] == 1:
            value=Pair.f_conv(query[2])
            t.update(query[1],value,1,1,len(data))
        if query[0] == 2:
            l,r=query[1:]
            p=t.query(l,r,1,1,len(data))
            result=p[0]+p[1]
            results.append(result)
    for i in results:
        print(i)
            


if __name__ == "__main__":
    main()