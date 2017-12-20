# Multi media database

## Requirements

Install OpenCv: [https://medium.com/@debugvn/installing-opencv-3-3-0-on-ubuntu-16-04-lts-7db376f93961](https://medium.com/@debugvn/installing-opencv-3-3-0-on-ubuntu-16-04-lts-7db376f93961)

## Install

```sh
git clone https://github.com/NTT-TNN/MMDB
cd MMDB/ImageSearch
workon cv
python manager runserver
open brower at : http://127.0.0.1:8000 to search by text (example :food,bus,person,horse,....)
open brower at : http://127.0.0.1:8000/photos to search by image (bag of word)
open brower at : http://127.0.0.1:8000/photos/sorting to search by iamge (sorting limit 100)



```
