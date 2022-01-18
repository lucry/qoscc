
#pragma once
#ifndef QOSCC_H_
#define QOSCC_H_
//#define PY_SSIZE_T_CLEAN

#define OK_SIGNAL 99999
#define NO_ACTION -11111
#define MAX_TIMESLOT 1000000

#include <iostream>
#include <pthread.h>
#include <math.h>

//Shared Memory ==> Communication with RL-Module -----*
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>


#include "../core/udt.h"
#include "../core/ccc.h"
#include "../core/common.h"
//#include "Python.h"

//PyObject* InitPythonThread();




/**
* class: QoSCC
**/
class QoSCC:public CCC
{
public:
    QoSCC();
    ~QoSCC();
    void init();
//    virtual int buildAgent();
    virtual void onACK(int32_t ack);
    virtual void onLoss(const int32_t*, int);
    virtual void onTimeout();
protected:
    void QoSCCUpdate();        //update the state and train the model
    void QoSCCReset();         //reset the algorithm parameters
    double ReqCompletion(double x, double y); //requirenmet completion ratio
    void setShareKey(int sharekey, int sharekey_rl);    //set share memory key
    void setTarget(double TargetBW, double TargetRTT, double WeightBW, double WeightRTT);   //set application requirement

    double getSendingRate();         //get sending rate by python API
//    double getPythonAPI(); //interact with python

    // parameters related to rate control
    int m_iRCInterval; //rate control interval
    uint64_t m_LastRCTime; //last rate control time
    double m_sending_rate; //sending rate
    int32_t m_iLastSndSeqNo; //maximum seq no sent out when last rate control

    //parameters related to machine learning
    bool m_isTrain; //if train the model
    int32_t m_iIDAction; //ack of the action
    int32_t m_iIDState; //ack of the state
    bool m_bfirstaction;
    bool m_bReset; // State of reset

    //parameters related to environment state
    int m_iMaxBWinInterval; //maximum bandwidth in current rate control interval
    int m_iMinRTTinInterval; //minimum RTT in current rate control interval
    int m_iTimeSlot; //time slot id;

    //parameters related to application requirements
    double m_dTargetThr; //target throughput,Mbps
    double m_dTargetRTT; //target RTT,ms
    double m_dwThr; //weight of throughput
    double m_dwRTT; //weight of RTT


    //parameters related to tansport protocol
    bool m_bSlowStart; //if it is slow start phase
    int m_iLastAck;             //last ACK time
    int m_WMax;               //maximum CWND window size
    int m_LastWMax;
    int m_issthresh;         //the CWND threshold for congestion avoid
    int cwnd_cnt;
    int ack_cnt;
    int max_cnt;
    int epoch_start;
    int origin_point;

    //parameters related to perf information
    const CPerfMon* m_QoSCCPerfInfo;
    uint64_t m_iLastTraceSend;
    int m_iLastTraceLoss;
    uint64_t m_iLastSampleTime;





    //parameters related to share memory
    key_t m_key; //key for share memory store state
    key_t m_key_rl; //keet for share memory store action
    int m_shmid; //share memory id store state
    int m_shmid_rl; //share memory id store action
    int m_shmem_size;
    char *m_shared_memory;
    char *m_shared_memory_rl;
    char m_message[1000];

    //parameters related to python API
//    PyObject *m_pModule;  //python file module
////    PyObject *m_pClass; //python class
//    PyObject *m_pAgent;  //instance of python class

    //parameters related to thread management
    pthread_mutex_t lockit;
    pthread_t init_python_thread;

};


#endif
