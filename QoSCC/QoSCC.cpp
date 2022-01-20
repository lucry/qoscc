#include "QoSCC.h"
#include "../core/core.h"

#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstring>
#include <unistd.h>

using std::cout;
using std::endl;


QoSCC::QoSCC() :
    m_iRCInterval(),
    m_LastRCTime(0),
    m_isTrain(true),
    m_bfirstaction(true),
    m_bReset(false),
    m_iMaxBWinInterval(),
    m_iMinRTTinInterval(),
    m_iTimeSlot(0),
    m_iLastAck(),
    m_WMax(),
    m_iLastTraceSend(0),
    m_iLastTraceLoss(0),
    m_LastWMax(),
    m_issthresh(),
    m_dTargetThr(),
    m_dTargetRTT(),
    m_dwThr(),
    m_dwRTT(),
    m_key(123456),
    m_key_rl(12345),
    m_shmid(NULL),
    m_shmid_rl(NULL),
    m_shmem_size(2048),
    m_shared_memory(NULL),
    m_shared_memory_rl(NULL)
//    m_pModule(NULL),
//    m_pAgent(NULL)
    {

}

QoSCC::~QoSCC(){
    shmdt(m_shared_memory);
    shmctl(m_shmid, IPC_RMID, NULL);
    shmdt(m_shared_memory_rl);
    shmctl(m_shmid_rl, IPC_RMID, NULL);
}

/**
 * initial function ---- init all parameters
 * */
void QoSCC::init(){
    m_iRCInterval = m_iRTT;
    m_LastRCTime = CTimer::getTime();
    m_iLastSampleTime = CTimer::getTime();
    // setACKTimer(0);
    setACKInterval(1);

    m_iMaxBWinInterval = 0;
    m_iMinRTTinInterval = 0;
    m_iLastAck = m_iSndCurrSeqNo;
    m_iLastSndSeqNo = m_iSndCurrSeqNo;
    m_iIDAction = -1;
    m_iIDState = -1;

    m_dTargetThr = 300; //Mbps
    m_dTargetRTT = 20; //ms
    m_dwThr = 0.5;
    m_dwRTT = 0.5;

    m_dCWndSize = 16;
    m_dPktSndPeriod = (2*(m_iMSS-28.0-16.0)*8.0) / m_dTargetThr;
    m_dMaxCWndSize = 60;

    // Setup shared memory
    if ((m_shmid = shmget(m_key, m_shmem_size, IPC_CREAT | 0666)) < 0)
    {
        printf("QoSCC: error getting shared memory id");
        exit(-1);
    }
    // Attached shared memory
    if ((m_shared_memory = (char*)shmat(m_shmid, NULL, 0)) == (char *) -1)
    {
        printf("QoSCC: error attaching shared memory id");
        exit(-1);
    }
   // Setup shared memory
    if ((m_shmid_rl = shmget(m_key_rl, m_shmem_size, IPC_CREAT | 0666)) < 0)
    {
        printf("QoSCC: error getting shared memory id");
        exit(-1);
    }
    // Attached shared memory
    if ((m_shared_memory_rl = (char*)shmat(m_shmid_rl, NULL, 0)) == (char *) -1)
    {
        printf("QoSCC: error attaching shared memory id");
        exit(-1);
    }

}

void QoSCC::setShareKey(int sharekey, int sharekey_rl){
    m_key = (key_t) sharekey;
    m_key_rl = (key_t) sharekey_rl;
}

void QoSCC::setTarget(double TargetBW, double TargetRTT, double WeightBW, double WeightRTT) {
    m_dTargetThr = TargetBW;
    m_dTargetRTT = TargetRTT;
    m_dwThr = WeightBW;
    m_dwRTT = m_dwRTT;
}

void QoSCC::QoSCCReset()
{
    m_dCWndSize = 16;
    m_dPktSndPeriod = 2*((m_iMSS-28.0-16.0)*8.0) / m_dTargetThr;
}

void QoSCC::onACK(int32_t ack){
    uint64_t currtime = CTimer::getTime();
    m_iMaxBWinInterval = std::max(m_iMaxBWinInterval, m_iBandwidth);
    m_iMinRTTinInterval = std::min(m_iMinRTTinInterval, m_iRTT);
    bool reset = false;
    // write state information from shared memory
//    std::cout << "Current state. RCinterval: " << currtime - m_LastRCTime << " ACK:" << m_iIDState << " " << m_iIDAction << "First time: " << m_bfirstaction << endl;
    if ((currtime - m_LastRCTime >= (uint64_t)m_iRTT && m_iIDState==m_iIDAction) || m_iIDState == -1){
        // reset QoSCC when learning failed
        double maxthr = m_iMaxBWinInterval * (m_iMSS-28-16) * 8.0 / 1000000.0; //Mbps, packet size - packet offload
        double minrtt = m_iMinRTTinInterval / 1000.0; //ms
        m_QoSCCPerfInfo = getPerfInfo();
        uint64_t current_send = m_QoSCCPerfInfo->pktSentTotal;
        int current_loss = m_QoSCCPerfInfo->pktSndLossTotal;
        double loss = (double) (current_loss-m_iLastTraceLoss) / (current_send-m_iLastTraceSend);
        double trace_rate = (double) (current_send-m_iLastTraceSend) * (m_iMSS-28.0-16.0) * 8.0 / (currtime-m_iLastSampleTime);
        m_sending_rate = (m_iMSS-28.0-16.0) * 8.0 * CSeqNo::seqlen(m_iLastSndSeqNo, m_iSndCurrSeqNo) / (currtime-m_LastRCTime);
        // std::cout << "Sending rate is: " << m_sending_rate << " " << CSeqNo::seqlen(m_iLastSndSeqNo, m_iSndCurrSeqNo) << " " << (currtime-m_LastRCTime) << endl;
        double reward = m_dwThr*ReqCompletion(m_sending_rate, m_dTargetThr)+m_dwRTT*ReqCompletion(m_dTargetRTT, minrtt);
        m_iIDState = (m_iIDState+1) % MAX_TIMESLOT;
//        std::cout << "Current state id is: " << m_iIDState << endl;
        if(!m_bReset){
            sprintf(m_message, "%d %.7f %.7f %.7f %.7f %.7f %.7f %.7f", m_iIDState, maxthr, minrtt, trace_rate,  loss, m_sending_rate, reward, m_dPktSndPeriod); //store state into share memory, throughput(MB/s), RTT(ms)
        }else{
            sprintf(m_message, "%d %.7f %.7f %.7f %.7f %.7f %.7f %.7f", m_iIDState, maxthr, minrtt, trace_rate,  loss, m_sending_rate, -1.0, -1.0); //store state into share memory, throughput(MB/s), RTT(ms)
        }
        std::cout << "UDT write state: " << m_message << endl;
        m_bReset = false;
        m_iLastSampleTime = currtime;
        m_iLastTraceSend = current_send;
        m_iLastTraceLoss = current_loss;
        m_iRCInterval = m_iRTT;
        memcpy(m_shared_memory, m_message, sizeof(m_message));
    }

    //read action information from shared memory
    double alpha = getSendingRate();
//    std::cout << "Get action. State: " << m_iIDState << " Action:" << m_iIDAction << endl;
    if(m_iIDState == m_iIDAction && currtime - m_LastRCTime >= (uint64_t)m_iRCInterval){
//        std::cout << "###############Before change: " << m_dPktSndPeriod << endl;
        if(alpha>0){
//              m_dCWndSize = m_dCWndSize * (1.0+alpha);
            m_dPktSndPeriod = m_dPktSndPeriod * (1.0+alpha);
        }
        else if(alpha<0 && alpha!=NO_ACTION){
//              m_dCWndSize = m_dCWndSize / (1.0-alpha);
            m_dPktSndPeriod = m_dPktSndPeriod / (1.0-alpha);
        }

    //    std::cout << "udt: After change: " << m_dPktSndPeriod << " " << alpha << endl;
        if((m_dPktSndPeriod < (0.1*(m_iMSS-28.0-16.0)*8.0)/m_dTargetThr) || (m_dPktSndPeriod > (10.0*(m_iMSS-28.0-16.0)*8.0)/m_dTargetThr)){
            std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&Reset QoSCC state. " << ack << " " << m_iMSS << " " << m_dTargetThr << " " << m_dPktSndPeriod << endl;
            QoSCCReset();
            m_bReset = true;
        }

//        m_sending_rate = (m_iMSS-28-16) * 8.0 / m_dPktSndPeriod; //Mbps
        m_dCWndSize = (double)m_iRTT / m_dPktSndPeriod+2;
//        m_dPktSndPeriod = m_dCWndSize / (m_iRTT + m_iRCInterval);
        m_iMaxBWinInterval = m_iBandwidth;
        m_iMinRTTinInterval = m_iRTT;
        m_LastRCTime = currtime;
        m_iLastSndSeqNo = m_iSndCurrSeqNo;
    }
//    else if (currtime - m_LastRCTime >= 10e6){
//        std::cout << "No action from agent for 2 seconds..." << endl;
//        std::cout << "m_iIDState: " << m_iIDState << " m_iIDAction: " << m_iIDAction << " currtime - m_LastRCTime: " << currtime - m_LastRCTime << " m_iRCInterval: " << m_iRCInterval << endl;
//        exit(0);
//    }



}

void QoSCC::onLoss(const int32_t*, int) {}

void QoSCC::onTimeout(){}

double QoSCC::ReqCompletion(double x, double y){
    if (x>=y && x>0 && y>0){
        return 1.0;
    }else{
        return x/(y+1e-7);
    }
}

double QoSCC::getSendingRate(){
    char *id_memory=NULL;
    char *action=NULL;
    char *save_ptr=NULL;
    double rate;
    id_memory = strtok_r(m_shared_memory_rl," ",&save_ptr);
    action = strtok_r(NULL," ",&save_ptr);
    
    // if(id_memory!=NULL && action!=NULL && atoi(id_memory)==(m_iIDAction+1.0)/MAX_TIMESLOT){
    if(id_memory!=NULL && action!=NULL && atoi(id_memory)!=m_iIDAction){
        rate = atof(action);
        m_iIDAction = atoi(id_memory);
         std::cout << "UDT get action: " << rate << " " << m_iIDAction << endl;
        return rate;
    }
    else{
//        std::cout << "No action." << *id_memory << " " << *action << " " << << " " << m_iIDAction << endl;
        return NO_ACTION;
    }
    
    
}

// double QoSCC::getSendingRate(){
//     char *id_memory;
//     char *action;
//     char *save_ptr;
//     double rate;
//     id_memory = strtok_r(m_shared_memory_rl," ",&save_ptr);
//     action = strtok_r(NULL," ",&save_ptr);
//     while (true){
//         if(id_memory!=NULL && action!=NULL && atoi(id_memory)!=m_iIDAction){
//             rate = atof(action);
//             m_iIDAction = atoi(id_memory);
//            std::cout << "UDT get action: " << rate << " " << m_iIDAction << endl;
//             return rate;
//         }
//         else{
//     //        std::cout << "No action." << endl;
//             // return NO_ACTION;
//             usleep(10);
//         }
//     }
    
// }



