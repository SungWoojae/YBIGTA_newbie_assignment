from lib import Trie
import sys


"""
TODO:
- 일단 Trie부터 구현하기
- main 구현하기

힌트: 한 글자짜리 자료에도 그냥 str을 쓰기에는 메모리가 아깝다...
"""


def main() -> None:
    names=[]
    iter=int(sys.stdin.readline())
    for _ in range(iter):
        name=sys.stdin.readline().strip()
        names.append(name)
    t:Trie=Trie()
    for j in names:
        t.push(j)
    
    print(t.num_of_root())

if __name__ == "__main__":
    main()