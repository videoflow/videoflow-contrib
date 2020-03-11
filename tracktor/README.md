# Traktor

This package provides the implementation of the paper **Tracking without bells and whistles** (Philipp Bergmann, [Tim Meinhardt](https://dvl.in.tum.de/team/meinhardt/), [Laura Leal-Taixe](https://dvl.in.tum.de/team/lealtaixe/)) [https://arxiv.org/abs/1903.05625].

It includes an updated version of Tracktor for Pytorch 1.3 with an improved object detector and a reid model.  You can use the tracker with your own object detector, or you can use it with the built-in detector.

## Installing

## Evaluation Results
This is how the pretrained models provided perform on the `MOT17` Challenge. 

```
********************* MOT17 TRAIN Results *********************
IDF1  IDP  IDR| Rcll  Prcn   FAR|   GT  MT   PT   ML|    FP    FN   IDs    FM|  MOTA  MOTP MOTAL
65.2 83.8 53.3| 63.1  99.2  0.11| 1638 550  714  374|  1732124291   903  1258|  62.3  89.6  62.6

********************* MOT17 TEST Results *********************
IDF1  IDP  IDR| Rcll  Prcn   FAR|   GT  MT   PT   ML|    FP    FN   IDs    FM|  MOTA  MOTP MOTAL
55.1 73.6 44.1| 58.3  97.4  0.50| 2355 498 1026  831|  8866235449  1987  3763|  56.3  78.8  56.7
```