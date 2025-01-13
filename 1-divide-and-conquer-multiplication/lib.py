from __future__ import annotations
import copy


"""
TODO:
- __setitem__ 구현하기
- __pow__ 구현하기 (__matmul__을 활용해봅시다)
- __repr__ 구현하기
"""


class Matrix:
    MOD = 1000

    def __init__(self, matrix: list[list[int]]) -> None:
        self.matrix = matrix

    @staticmethod
    def full(n: int, shape: tuple[int, int]) -> Matrix:
        return Matrix([[n] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        return self.matrix[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        '''
        tuple을 통해 받은 좌표에 있는 값을 인자로 받은 값으로 변환해주는 메소드
            - 인자
                - key (tuple[int,int]) : 바꾸고 싶은 좌표
                - value (int) : 새롭게 넣고 싶은 값
            - 출력 : 없음 (None)
        '''
        self.matrix[key[0]][key[1]]=value % Matrix.MOD

    def __matmul__(self, matrix: Matrix) -> Matrix:
        x, m = self.shape
        m1, y = matrix.shape
        assert m == m1

        result = self.zeros((x, y))

        for i in range(x):
            for j in range(y):
                for k in range(m):
                    result[i, j] += self[i, k] * matrix[k, j]

        return result

    def __pow__(self, n: int) -> Matrix:
        '''
        __matmul__을 @를 통해 호출하여 분할 정복을 통한 객체의 거듭제곱을 계산하는 매직 메소드
        
            - 인자
                - n (int) : 몇번 거듭제곱을 할지 나타내는 정수값 (1<=n<=100,000,000,000)
            - 출력 : Matrix 객체
        '''
        assert self.shape[0]==self.shape[1]

        result = Matrix.eye(self.shape[0])
        base = self

        while n > 0:
            if n % 2 == 1:
                result = result @ base
            base = base @ base
            n //= 2

        return result
        

    def __repr__(self) -> str:
        '''
        객체의 문자열 표현을 반환하는 매직 메서드.
        행렬 데이터를 문자열로 변환하여 반환.
        또한, 각 요소에 대해 MOD1000 연산 수행.

            - 출력 : str
        '''
        return "\n".join(
            " ".join(str(self[i, j]) for j in range(self.shape[1]))
            for i in range(self.shape[0])
        )