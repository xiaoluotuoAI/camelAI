#include <stdio.h>
#include <algorithm>
#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

static int classifier(const cv::Mat& bgr, std::vector<float>& pre){
    ncnn::Net mlp;

    mlp.load_param("ncnn.param");
    mlp.load_model("ncnn.bin");

    ncnn::Mat in;
    ncnn::Mat out;
    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data,ncnn::Mat::PIXEL_BGR,bgr.cols,bgr.rows)

    for(int i =0;i<bgr.cols*bgr.rows;i++){
        printf("%dth: %d",i,bgr.data[i]);
    }


}

int main(int argc,char** argv)
{
    if(argc !=2)
        {
            fprintf(stderr,"Usage: %s [imagepath]\n",argv[0]);
            return -1;
        }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath,CV_LOAD_IMAGE_COLOR);

    if(m.empty())
    {
        fprintf(stderr,"cv::imread %s failed\n",imagepath);
        return -1;
    }

    std::vector<float> cls_scores;

    classifier();

    fprintf("OK!");


}




