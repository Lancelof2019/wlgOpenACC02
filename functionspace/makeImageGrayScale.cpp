#include "../headerspace/WatershedAlg.h"
using namespace cv;

Mat WatershedAlg::makeImageGrayScale(Mat &image) {
      
      Mat grayScale(image.rows, image.cols, CV_8UC1, Scalar::all(0));
  
/*        double gray=0.0,gray1=0.0;
     
      #pragma omp parallel for  //parallel for simd  
      for(int i = 0; i < image.rows; i++) {
          #pragma omp parallel for  
            for(int j = 0; j < image.cols; j++) {
                 gray = 0.21 * image.at<cv::Vec3b>(i,j)[0] +
                              0.72 * image.at<cv::Vec3b>(i,j)[1] +
                              0.07 * image.at<cv::Vec3b>(i,j)[2];

                grayScale.at<uchar>(i,j) = (uchar)gray;
            }
        }
 */











        auto *startImg1=image.data;
        auto *startgrayScale=grayScale.data;
        int imgrows1=image.rows;
        int imgcols1=image.cols;
        #pragma acc enter data copyin(startImg1[:3*imgrows1*imgcols1],startgrayScale[:imgrows1*imgcols1])
        #pragma acc parallel loop collapse(2) default(present)

        for(int i = 0; i < imgrows1; i++) {

            for(int j = 0; j < imgcols1; j++) {

             double gray=0.21*startImg1[i*imgcols1*3+j*3+0] + 0.72*startImg1[i*imgcols1*3+j*3+1]+0.07*startImg1[i*imgcols1*3+j*3+2];
             startgrayScale[i*imgcols1+j]=(uchar)gray;

            }
        }

        #pragma acc update self(startgrayScale[:imgrows1*imgcols1])
        #pragma acc exit data delete(startImg1[:3*imgrows1*imgcols1],startgrayScale[:imgrows1*imgcols1])

   
       
        cv::GaussianBlur(grayScale, grayScale, Size(3,5), 3,4);
       
        image.release();
        return grayScale;
    }

