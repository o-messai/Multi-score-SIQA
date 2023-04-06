
# End-to-end deep multi-score model for No-reference stereoscopic image quality assessment </br>
## Multi-score-SIQA </br>

This is the implementation code for our paper published in @ICIP2022 conference: </br>
"End-to-end deep multi-score model for No-reference stereoscopic image quality assessment" </br>
link: https://arxiv.org/abs/2211.01374

## Online web-application

You can use our proposed model online at: https://oussama-messai.com/iqa-stereo </br>
The model will run on the browser once it's loaded (on-edge mode), therefore the run-time depends on your machine. </br>
It should work with any image resolution greater than 32 x 32 pixels, and it supports all image formats (e.g., BMP, PNG, JPG, GIF ...etc), please note that the app does not record or save any user's data/images but a feedback is much appreciated.

![image](https://github.com/o-messai/multi-score-SIQA/blob/main/results/online.png?raw=true)

## Requirments

- torch </br>
- scipy </br>
- numpy </br> 
- PIL </br>


## Citation

            @proceeding{messai2022-end-to-end,
            title={End-to-end deep multi-score model for No-reference stereoscopic image quality assessment},
            author={Messai, Oussama and Chetouani, Aladine},
            conference={ICIP2022},
            year={2022},
            publisher={IEEE}}

####
N.B: Please note that the code will be cleaned/commented in further versions.
