#include "cv.h"
#include "highgui.h"

#include <opencv2/opencv.hpp>
#include <istream>
#include "JpKalmanFilter.h"

using namespace cv;
using namespace std;

void print(Mat temp)
{
    cout << temp << endl;
}

int main()
{
    // time interval
    float interval = 100; // unit : ms
    JpKalmanFilter kalman(4, 2);
    //观测矩阵
    kalman.measurementMatrix = (Mat_<float>(2, 4) << 1, 0, 0, 0,
                                                     0, 1, 0, 0);

    //观测方差噪声
    setIdentity(kalman.measurementNoiseCov, Scalar(10));


    //转移矩阵
    kalman.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0,
                                                    0, 1, 0, 1,
                                                    0, 0, 1, 0,
                                                    0, 0, 0, 1);
    //过程噪声
    setIdentity(kalman.processNoiseCov, Scalar(1e-1));

    //后验方差
    setIdentity(kalman.errorCovPost, Scalar(1));

    namedWindow("Kalman");

    Mat measure(2,1,CV_32FC1);
    Mat trueState;
    trueState = (Mat_<float>(4, 1) << 40, 40, 10, 10);

    // Kalman的初始化状态可以任意赋值，但是尽量和真实位置相近
    RNG rng;
    //kalman.statePost = trueState.clone();
    Mat initState(4, 1, CV_32FC1);
    rng.fill(initState, RNG::NORMAL, 0, 1);
    kalman.statePost = initState.clone();

    while(1)
    {
        Mat image(450, 600, CV_8UC3, Scalar(0, 0, 0)); //初始绘图
        rectangle(image,Point(30,30),Point(570,420),Scalar(255,255,255),2);//绘制目标弹球的“撞击壁”

        //观测数据模拟，观测数据符合相应的高斯分布
        Mat measure_error(2, 1, CV_32FC1);
        rng.fill(measure_error,RNG::NORMAL,0, sqrt(kalman.measurementNoiseCov.at<float>(0,0)));
        measure = kalman.measurementMatrix*trueState + measure_error;

        //Kalman Filter赋观测值
        kalman.observe(measure);
        //更新：模型预测,并结合观测加权给出最终结果
        Mat state = kalman.update();
        cout << "State " ;print(state);
        cout << "Measure "; print(measure);
        circle(image, Point(state.at<float>(0,0),state.at<float>(0,1)), 4, Scalar(255, 255, 0), 2); // 青色 - kalman filter预测位置
        circle(image, Point(trueState.at<float>(0,0), trueState.at<float>(0,1)), 4, Scalar(255, 255, 255), 2); //白色 - 真实位置
        circle(image, Point(measure.at<float>(0,0), measure.at<float>(0,1)), 4, Scalar(0, 255, 0), 2); //绿色 - 观测位置

        //假设真实运动为严格匀速运动
        trueState = kalman.transitionMatrix*trueState;
        print(trueState);

        //假设真实运动为匀速运动,允许有误差，误差为高斯
        //Mat trueState_error(4, 1, CV_32FC1);
        //rng.fill(trueState_error, RNG::NORMAL, 0, sqrt(kalman.processNoiseCov.at<float>(0,0)));
        //trueState = kalman.transitionMatrix*trueState + trueState_error;



        if(trueState.at<float>(0,0)<30){  //当撞击到反弹壁时，对应轴方向取反
            trueState.at<float>(0,2) *= -1;
        }
        if(trueState.at<float>(0,0)>570){
            trueState.at<float>(0,2) *= -1;
        }
        if(trueState.at<float>(0,1)<30){
            trueState.at<float>(0,3) *= -1;
        }
        if(trueState.at<float>(0,1)>420){
            trueState.at<float>(0,3) *= -1;
        }

        imshow("Kalman", image);
        if(waitKey(interval) == '27')
        {
            break;
        }
    }
    return 0;

}

/*opencv 实现 Kalman Filter*
 *注意：以下算法中，cvKalmanPredict仅实现了预测步骤，并没有根据观测进行更新
 * 更新命令为cvKalmanUpdateByMeasurement*/
//int main ()
//{
//    cvNamedWindow("Kalman",1);
//    CvRandState random;//创建随机
//    cvRandInit(&random,0,1,-1,CV_RAND_NORMAL);
//    IplImage * image= cvCreateImage(cvSize(600,450),8,3);
//    CvKalman * kalman=cvCreateKalman(4,2,0);//状态变量4维，x、y坐标和在x、y方向上的速度，测量变量2维，x、y坐标
//
//    CvMat * xK=cvCreateMat(4,1,CV_32FC1);//初始化状态变量，坐标为（40,40），x、y方向初速度分别为10、10
//    xK->data.fl[0]=40.;
//    xK->data.fl[1]=40;
//    xK->data.fl[2]=10;
//    xK->data.fl[3]=10;
//
//    const float F[]={1,0,1,0,0,1,0,1,0,0,1,0,0,0,0,1};//初始化传递矩阵 [1  0  1  0]
//                                                      //               [0  1  0  1]
//                                                      //               [0  0  1  0]
//                                                      //               [0  0  0  1]
//    memcpy(kalman->transition_matrix->data.fl,F,sizeof(F));
//
//
//
//    CvMat * wK=cvCreateMat(4,1,CV_32FC1);//过程噪声
//    cvZero(wK);
//
//    CvMat * zK=cvCreateMat(2,1,CV_32FC1);//测量矩阵2维，x、y坐标
//    cvZero(zK);
//
//    CvMat * vK=cvCreateMat(2,1,CV_32FC1);//测量噪声
//    cvZero(vK);
//
//    cvSetIdentity( kalman->measurement_matrix, cvScalarAll(1) );//初始化测量矩阵H=[1  0  0  0]
//                                                                //                [0  1  0  0]
//    cvSetIdentity( kalman->process_noise_cov, cvScalarAll(1e-1) );/*过程噪声____设置适当数值，
//                                                                    增大目标运动的随机性，
//                                                                    但若设置的很大，则系统不能收敛，
//                                                                    即速度越来越快*/
//    cvSetIdentity( kalman->measurement_noise_cov, cvScalarAll(10) );/*观测噪声____故意将观测噪声设置得很大，
//                                                                    使之测量结果和预测结果同样存在误差*/
//    cvSetIdentity( kalman->error_cov_post, cvRealScalar(1) );/*后验误差协方差*/
//    cvRand( &random, kalman->state_post );
//
//    CvMat * mK=cvCreateMat(1,1,CV_32FC1);  //反弹时外加的随机化矩阵
//
//
//    while(1){
//        cvZero( image );
//        cvRectangle(image,cvPoint(30,30),cvPoint(570,420),CV_RGB(255,255,255),2);//绘制目标弹球的“撞击壁”
//        const CvMat *yK  =cvKalmanPredict(kalman,0);//计算预测位置
//        printf("%f_____%f\n",yK->data.fl[0],yK->data.fl[1]);
//        //printf("%f_____%f\n",yK->data.fl[0],yK->data.fl[1]);
//        cvRandSetRange( &random, 0, sqrt( kalman->measurement_noise_cov->data.fl[0] ), 0 );
//        cvRand( &random, vK );//设置随机的测量误差
//        cvMatMulAdd( kalman->measurement_matrix, xK, vK, zK );//zK=H*xK+vK
//        cvCircle(image,cvPoint(cvRound(CV_MAT_ELEM(*xK,float,0,0)),cvRound(CV_MAT_ELEM(*xK,float,1,0))),
//            4,CV_RGB(255,255,255),2);//白圈，真实位置
//        cvCircle(image,cvPoint(cvRound(CV_MAT_ELEM(*yK,float,0,0)),cvRound(CV_MAT_ELEM(*yK,float,1,0))),
//            4,CV_RGB(0,255,0),2);//绿圈，预估位置
//        cvCircle(image,cvPoint(cvRound(CV_MAT_ELEM(*zK,float,0,0)),cvRound(CV_MAT_ELEM(*zK,float,1,0))),
//            4,CV_RGB(0,0,255),2);//蓝圈，观测位置
//
//        cvRandSetRange(&random,0,sqrt(kalman->process_noise_cov->data.fl[0]),0);
//        cvRand(&random,wK);//设置随机的过程误差
//        cvMatMulAdd(kalman->transition_matrix,xK,wK,xK);//xK=F*xK+wK
//
//        if(cvRound(CV_MAT_ELEM(*xK,float,0,0))<30){  //当撞击到反弹壁时，对应轴方向取反外加随机化
//            cvRandSetRange( &random, 0, sqrt(1e-1), 0 );
//            cvRand( &random, mK );
//            xK->data.fl[2]=10+CV_MAT_ELEM(*mK,float,0,0);
//        }
//        if(cvRound(CV_MAT_ELEM(*xK,float,0,0))>570){
//            cvRandSetRange( &random, 0, sqrt(1e-2), 0 );
//            cvRand( &random, mK );
//            xK->data.fl[2]=-(10+CV_MAT_ELEM(*mK,float,0,0));
//        }
//        if(cvRound(CV_MAT_ELEM(*xK,float,1,0))<30){
//            cvRandSetRange( &random, 0, sqrt(1e-1), 0 );
//            cvRand( &random, mK );
//            xK->data.fl[3]=10+CV_MAT_ELEM(*mK,float,0,0);
//        }
//        if(cvRound(CV_MAT_ELEM(*xK,float,1,0))>420){
//            cvRandSetRange( &random, 0, sqrt(1e-3), 0 );
//            cvRand( &random, mK );
//            xK->data.fl[3]=-(10+CV_MAT_ELEM(*mK,float,0,0));
//        }
//
//
//
//        cvShowImage("Kalman",image);
//
//        cvKalmanCorrect( kalman, zK );
//
//
//        if(cvWaitKey(100)=='e'){
//            break;
//        }
//    }
//
//
//    cvReleaseImage(&image);/*释放图像*/
//    cvDestroyAllWindows();
//}
