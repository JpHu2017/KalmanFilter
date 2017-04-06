//
// Created by cvrsg on 17-3-31.
//

#ifndef KALMANFILTER_JPKALMANFILTER_H
#define KALMANFILTER_JPKALMANFILTER_H

#include <opencv2/core.hpp>

using namespace cv;

class JpKalmanFilter {
public:
    JpKalmanFilter(int transitionParams, int measureParams, int controlParams = 0);

    Mat update();

    void observe(const Mat& measure);
    void control(const Mat& control);

    Mat statePre;           //!< predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k) -- B is usually 0
    Mat statePost;          //!< corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-C*x'(k))
    Mat transitionMatrix;   //!< state transition matrix (A)
    Mat controlMatrix;      //!< control matrix (B) (not used if there is no control)
    Mat measurementMatrix;  //!< measurement matrix (C)
    Mat processNoiseCov;    //!< process noise covariance matrix (Q, Tao)
    Mat measurementNoiseCov;//!< measurement noise covariance matrix (R, Sigma)
    Mat errorCovPre;        //!< priori error estimate covariance matrix (P(k-1)): P(k-1)=A*V(k-1)*At + Q)*/
    Mat gain;               //!< Kalman gain matrix (K(k)): K(k)=P(k-1)*Ht*inv(C*P(k-1)*Ct+R)
    Mat errorCovPost;       //!< posteriori error estimate covariance matrix (V(k)): V(k)=(I-K(k)*C)*P(k-1)

private:
    void predict(); //预测
    void process(); //计算Kalman Filter中的一些变量

private:
    int m_transitionParams;
    int m_measureParams;
    int m_controlParams;

    Mat m_measure;
    Mat m_control;

    bool m_init_flag;
};


#endif //KALMANFILTER_JPKALMANFILTER_H
