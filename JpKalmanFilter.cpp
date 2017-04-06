//
// Created by cvrsg on 17-3-31.
//

#include "JpKalmanFilter.h"
#include <iostream>

using namespace std;
using namespace cv;


JpKalmanFilter::JpKalmanFilter(int transitionParams, int measureParams, int controlParams)
    : m_transitionParams(transitionParams), m_measureParams(measureParams), m_controlParams(controlParams)
{
    if(m_controlParams)
    {
        controlMatrix.create(m_transitionParams, m_controlParams, CV_32FC1);
    }
    transitionMatrix.create(m_transitionParams, m_transitionParams, CV_32FC1);
    processNoiseCov.create(m_transitionParams, m_transitionParams, CV_32FC1); // Tao

    measurementMatrix.create(m_measureParams, m_transitionParams, CV_32FC1);
    measurementNoiseCov.create(m_measureParams, m_measureParams, CV_32FC1); // Sigma

    errorCovPre.create(m_transitionParams, m_transitionParams, CV_32FC1);
    errorCovPost.create(m_transitionParams, m_transitionParams, CV_32FC1);

    m_init_flag = true;
}

Mat JpKalmanFilter::update() {
    process();
    predict(); //根据模型预测
    statePost = statePre + gain * (m_measure - measurementMatrix*statePre); //根据模型和观测修正
    return statePost;
}

void JpKalmanFilter::observe(const Mat &measure) {
    if(measure.rows == m_measureParams && measure.cols == 1)
    {
        m_measure = measure.clone();
    }
    else
    {
        cout << "Measure Variable is invalid!" << endl;
    }

}

void JpKalmanFilter::predict() {
    //不含控制变量时
    if(m_controlParams == 0)
    {
        //根据KF公式推导，初始化的statePost和之后时刻的稍有不同
        //但由于KF自己会不断修正，所以初始化采用和之后时刻一样的公式求解，影响不大
        //而且，虽说初始化状态（statePost）需要在真值附近，但实际上，这里随机给个值，如rnd(0)-0附近随机值也影响不大

//        if(m_init_flag) {
//            statePre = statePost.clone();
//            m_init_flag = false;
//        }
//        else
        {
            statePre = transitionMatrix*statePost;
        }
    }
    else
    {
        statePre = transitionMatrix*statePost + controlMatrix*m_control;
    }

}

void JpKalmanFilter::process() {
    errorCovPre = transitionMatrix*errorCovPost*transitionMatrix.t() + processNoiseCov;
    //Mat temp;
    //temp = measurementNoiseCov + measurementMatrix*errorCovPre*measurementMatrix.t();
    gain = errorCovPre*measurementMatrix.t()*(measurementNoiseCov + measurementMatrix*errorCovPre*measurementMatrix.t()).inv();
    Mat I(m_transitionParams, m_transitionParams, CV_32FC1, Scalar(0));
    setIdentity(I, Scalar(1));
    errorCovPost = (I - gain*measurementMatrix) * errorCovPre;
}

void JpKalmanFilter::control(const Mat &control) {
    m_control = control.clone();
}










