from lib import Trie
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