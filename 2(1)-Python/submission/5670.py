from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable
from math import factorial

"""
TODO:
- Trie.push 구현하기
- (필요할 경우) Trie에 추가 method 구현하기
"""


T = TypeVar("T")


@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: list[int] = field(default_factory=lambda: [])
    is_end: bool = False


class Trie(list[TrieNode[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.append(TrieNode(body=None))

    def push(self, seq: Iterable[T]) -> None:
        """
        seq: T의 열 (list[int]일 수도 있고 str일 수도 있고 등등...)

        action: trie에 seq을 저장하기
        """
        current_node_idx=0
        for i in seq:
            found=False
            for j in self[current_node_idx].children:
                if i==self[j].body:
                    found=True
                    current_node_idx=j
                    break
            if not found:
                new_node_idx=len(self)
                self.append(TrieNode(body=i))
                self[current_node_idx].children.append(new_node_idx)
                current_node_idx=new_node_idx
        self[current_node_idx].is_end=True

    def num_of_root(self) -> int:
        '''
        이름을 나열할 수 있는 순서의 개수를 구하는 함수.
        모든 노드의 자식 노드의 수, 다시 말해 children attribute의 크기를 곱하면 구할 수 있다.
            - 출력 (int) : self의 가능한 단어 나열 순서의 수
        '''
        num=1
        for node in self:
            num*=factorial(len(node.children))
        return num


import sys


"""
TODO:
- 일단 Trie부터 구현하기
- count 구현하기
- main 구현하기
"""


def count(trie: Trie, query_seq: str) -> int:
    """
    trie - 이름 그대로 trie
    query_seq - 단어 ("hello", "goodbye", "structures" 등)

    returns: query_seq의 단어를 입력하기 위해 버튼을 눌러야 하는 횟수
    """
    pointer = 0
    cnt = 0

    for element in query_seq:
        if len(trie[pointer].children) > 1 or trie[pointer].is_end:
            cnt += 1

        new_index = 0 # 구현하세요!
        for i in trie[pointer].children:
            if trie[i].body==element:
                new_index=i
                break
        pointer = new_index

    return cnt + int(len(trie[0].children) == 1)


def main() -> None:
    lines = [line.strip() for line in sys.stdin.readlines()]
    line_num=1
    while line_num<=len(lines):
        num=int(lines[line_num-1])
        words=lines[line_num:line_num+num]
        t:Trie=Trie()
        for j in words:
            t.push(j)
        ans=0
        for w in words:
            ans+=count(t,w)
        rounded_ans=f"{round(ans/num,2):.2f}"
        print(rounded_ans)
        line_num+=(num+1)
     


if __name__ == "__main__":
    main()