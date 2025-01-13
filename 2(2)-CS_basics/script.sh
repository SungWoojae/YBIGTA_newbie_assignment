#!/bin/bash

# miniconda가 존재하지 않을 경우 설치
if ! command -v conda &> /dev/null; then
    echo "Miniconda is not installed. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda init
    source ~/.bashrc
fi

# Conda 환경 생성 및 활성화
if conda env list | grep -q "myenv"; then
    echo "Activating existing conda environment: myenv"
    eval "$(conda shell.bash hook)"
    conda activate myenv
else
    echo "Creating and activating new conda environment: myenv"
    eval "$(conda shell.bash hook)"
    conda create -y -n myenv python=3.9
    conda activate myenv
fi

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "가상환경 활성화: 성공"
else
    echo "가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
pip install -r requirements.txt

# Submission 폴더 파일 실행
cd submission || { echo "submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    filename=$(basename -- "$file" .py)
    echo "Running $file..."
    # 파일 실행: input 디렉터리의 입력 파일을 읽고 output 디렉터리에 저장
    python "$file" < ../input/"${filename}_input" > ../output/"${filename}_output" || {
        echo "Error executing $file"
        exit 1
    }
    echo "Output saved to ../output/${filename}_output"
done

# mypy 테스트 실행
echo "Running mypy type checking..."
pip install mypy
mypy . || { echo "mypy 테스트 실패"; exit 1; }

# 가상환경 비활성화
echo "Deactivating conda environment..."
conda deactivate

echo "모든 작업이 성공적으로 완료되었습니다!"