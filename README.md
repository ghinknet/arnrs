# arnrs

Aircraft Registration Number Recognition System

## Installation

1. Install [Pytorch Engine](https://pytorch.org/get-started/locally/) (CUDA, ROCm or CPU) with anaconda

2. Clone Detectron2 Framework from Github with follow command:

    `git clone https://github.com/facebookresearch/detectron2.git`

3. Run Detectron2 Installtion with follow command:

    `python -m pip install -e detectron2 -i https://pypi.tuna.tsinghua.edu.cn/simple`

4. Install [PaddlePaddle Engine](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/windows-conda.html) (CUDA, ROCm or CPU) with anaconda

5. Install easyocr package with follow command:

    `pip install easyocr -i https://pypi.tuna.tsinghua.edu.cn/simple`

6. Install paddleocr package with follow command:

    `pip install paddleocr -i https://pypi.tuna.tsinghua.edu.cn/simple`

7. To solve opencv version issue, reinstall opencv and opencv-headless with follow commands (There may be some errors happen, you can ignore it)

    `pip install opencv-python==4.2.0.34 -i https://pypi.tuna.tsinghua.edu.cn/simple`

    `pip install opencv-python-headless==4.2.0.34 -i https://pypi.tuna.tsinghua.edu.cn/simple`

8. Replace packages rely in paddleocr from "tools.infer" to "paddleocr.tools.infer" with text editor, it's a bug from paddleocr

9. Finally enjoy it :)