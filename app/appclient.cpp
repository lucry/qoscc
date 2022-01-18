#ifndef WIN32
   #include <unistd.h>
   #include <cstdlib>
   #include <cstring>
   #include <netdb.h>
#else
   #include <winsock2.h>
   #include <ws2tcpip.h> 
   #include <wspiapi.h>
#endif
#include <iostream>
#include <udt.h>
#include <pthread.h>
#include <unistd.h>

//Shared Memory ==> Communication with RL-Module -----*
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <ctime>

#include "cc.h"
#include "test_util.h"
#include "../QoSCC/QoSCC.h"

#define INIT_AGENT 88888

using namespace std;

#ifndef WIN32
void* monitor(void*);
#else
DWORD WINAPI monitor(LPVOID);
#endif

struct Sh_Memory{
    int sh_key;
    int sh_key_rl;
};

void* PYThread(void *arg){
   Sh_Memory sh_mem = *(Sh_Memory *) arg;
   int sh_key = sh_mem.sh_key;
   int sh_key_rl = sh_mem.sh_key_rl;
   cout << "Appclient: Start building python agent..." << endl;
   char cmd[100];
//   cout << "cmd is: " << endl;
   sprintf(cmd, "python ../QoSCC/py_agent.py --sh_key=%d --sh_key_rl=%d", sh_key, sh_key_rl);
//   cout << cmd << endl;
   system(cmd);
}

int main(int argc, char* argv[])
{
   if (argc != 3 || 0 == atoi(argv[2])) {
      cout << "usage: " << argv[0] << " receiver_ip receiver_port" << endl;
      return 0;
   }

   Sh_Memory sh_mem;
   int shmid = -1;
   int shmid_rl = -1;
   char *shared_memory;
   char *shared_memory_rl;
   double target_bw = 100;
   double target_rtt = 10;
   double w_thr = 0.5;
   double w_rtt = 0.5;
   /**
   * build agent for QoSCC
   **/
   while(shmid < 0 || shmid_rl<0){
       sh_mem.sh_key = 123456+rand();
       sh_mem.sh_key_rl = 12345+rand();
       // Setup shared memory
       if ((shmid = shmget((key_t) sh_mem.sh_key, 2048, IPC_CREAT | 0666)) < 0)
       {
           cout << "Appclient: error getting shared memory id" << endl;
           return 0;
       }
       if ((shmid_rl = shmget((key_t) sh_mem.sh_key_rl, 2048, IPC_CREAT | 0666)) < 0)
       {
           cout << "Appclient: error getting shared memory id" << endl;
           return 0;
       }
   }

    // Attached shared memory
    if ((shared_memory = (char*)shmat(shmid, NULL, 0)) == (char *) -1)
    {
        printf("Appclient: error attaching shared memory id");
        return 0;
    }
    if ((shared_memory_rl = (char*)shmat(shmid_rl, NULL, 0)) == (char *) -1)
    {
        printf("Appclient: error attaching shared memory id");
        return 0;
    }
    //Clear shared memory
    memset(shared_memory, 0, 2048);
    memset(shared_memory_rl, 0, 2048);

   pthread_t py_thread;
   pthread_create(&py_thread, NULL, PYThread, &sh_mem);
   bool initial_agent = false;
   char *save_ptr;
   int error_cnt = 0;
   char *init_sig;
   while(!initial_agent){
        error_cnt++;
        init_sig = strtok_r(shared_memory_rl, " ", &save_ptr);
        if (init_sig == NULL or atoi(init_sig) !=INIT_AGENT){
            if(error_cnt > 60){
                cout << "Build agent failed in 1 minite..." << endl;
                return 0;
            }
            cout << "Building agent... (" << error_cnt << "s >> 60s)" << endl;
            if (init_sig != NULL){
                cout << "Something others in shared memory now: " << init_sig << endl;
            }
            usleep(1000000);
        }else{
            cout << "Build agent succeed!" <<endl;
            initial_agent = true;
        }
   }
//       shmdt(shared_memory);
//       shmctl(shmid, IPC_RMID, NULL);
//       shmdt(shared_memory_rl);
//       shmctl(shmid_rl, IPC_RMID, NULL);



   // Automatically start up and clean up UDT module.
   UDTUpDown _udt_;
   struct addrinfo hints, *local, *peer;
   memset(&hints, 0, sizeof(struct addrinfo));

   hints.ai_flags = AI_PASSIVE;
   hints.ai_family = AF_INET;
   hints.ai_socktype = SOCK_STREAM;
   //hints.ai_socktype = SOCK_DGRAM;

   if (0 != getaddrinfo(NULL, "9000", &hints, &local))
   {
      cout << "incorrect network address.\n" << endl;
      return 0;
   }

   UDTSOCKET client = UDT::socket(local->ai_family, local->ai_socktype, local->ai_protocol);

   // UDT Options

//        cout << "this is qoscc scheme" << endl;
    UDT::setsockopt(client, 0, UDT_CC, new CCCFactory<QoSCC>, sizeof(CCCFactory<QoSCC>));
    UDT::setsockopt(client, 0, UDT_SHAREKEY, new int(sh_mem.sh_key), sizeof(int));
    UDT::setsockopt(client, 0, UDT_SHAREKEYRL, new int(sh_mem.sh_key_rl), sizeof(int));
    UDT::setsockopt(client, 0, UDT_TARGETBW, new double(target_bw), sizeof(double));
    UDT::setsockopt(client, 0, UDT_TARGETRTT, new double(target_rtt), sizeof(double));
    UDT::setsockopt(client, 0, UDT_WEIGHTBW, new double(w_thr), sizeof(double));
    UDT::setsockopt(client, 0, UDT_WEIGHTRTT, new double(w_rtt), sizeof(double));


   //UDT::setsockopt(client, 0, UDT_MSS, new int(9000), sizeof(int));
   //UDT::setsockopt(client, 0, UDT_SNDBUF, new int(10000000), sizeof(int));
   //UDT::setsockopt(client, 0, UDP_SNDBUF, new int(10000000), sizeof(int));
//   UDT::setsockopt(client, 0, UDT_MAXBW, new int64_t(125000000), sizeof(int));

   // Windows UDP issue
   // For better performance, modify HKLM\System\CurrentControlSet\Services\Afd\Parameters\FastSendDatagramThreshold
   #ifdef WIN32
      UDT::setsockopt(client, 0, UDT_MSS, new int(1052), sizeof(int));
   #endif

   // for rendezvous connection, enable the code below
   /*
   UDT::setsockopt(client, 0, UDT_RENDEZVOUS, new bool(true), sizeof(bool));
   if (UDT::ERROR == UDT::bind(client, local->ai_addr, local->ai_addrlen))
   {
      cout << "bind: " << UDT::getlasterror().getErrorMessage() << endl;
      return 0;
   }
   */

   freeaddrinfo(local);

   if (0 != getaddrinfo(argv[1], argv[2], &hints, &peer))
   {
      cout << "incorrect server/peer address. " << argv[1] << ":" << argv[2] << endl;
      return 0;
   }

   // connect to the server, implict bind
   if (UDT::ERROR == UDT::connect(client, peer->ai_addr, peer->ai_addrlen))
   {
      cout << "connect: " << UDT::getlasterror().getErrorMessage() << endl;
      return 0;
   }

   freeaddrinfo(peer);

   // using CC method
//   cubic* cchandle = NULL;
   int temp;
    QoSCC* cchandle = NULL;
    UDT::getsockopt(client, 0, UDT_CC, &cchandle, &temp);

   int size = 100000;
   char* data = new char[size];

   #ifndef WIN32
      pthread_create(new pthread_t, NULL, monitor, &client);
   #else
      CreateThread(NULL, 0, monitor, &client, 0, NULL);
   #endif


   for (int i = 0; i < 1000000; i ++)
   {
      int ssize = 0;
      int ss;
      while (ssize < size)
      {
         if (UDT::ERROR == (ss = UDT::send(client, data + ssize, size - ssize, 0)))
         {
            cout << "send:" << UDT::getlasterror().getErrorMessage() << endl;
            break;
         }

         ssize += ss;
      }

      if (ssize < size)
         break;
   }

   UDT::close(client);
   delete [] data;
   return 0;
}

#ifndef WIN32
void* monitor(void* s)
#else
DWORD WINAPI monitor(LPVOID s)
#endif
{
   UDTSOCKET u = *(UDTSOCKET*)s;

   UDT::TRACEINFO perf;

   // ofstream tracelog;


   // ofstream sendRateLog;
   // remove("/home/QoSCC/plot/sendRate-log.txt");
   // sendRateLog.open("/home/QoSCC/plot/sendRate-log.txt", ios::out | ios::app);
   // sendRateLog << "[";

   // ofstream cwndLog;
   // remove("/home/QoSCC/plot/scwnd-log.txt");
   // cwndLog.open("/home/QoSCC/plot/cwnd-log.txt", ios::out | ios::app);
   // cwndLog << "[";

   ofstream clientLog;
   remove("/home/QoSCC/plot/client-log.txt");
//   sprintf((char)file, "/home/QoSCC/plot/statistics/client-log-%s.txt", )
   clientLog.open("/home/QoSCC/plot/client-log.txt", ios::out | ios::app);
   clientLog << "CurrentTime(s)\tSendRate(Mb/s)\tRTT(ms)\tCWnd\tPktSndPeriod(us)\tRecvACK\tRecvNAK\tPktSndLoss\tPktSent\tPktRetrans" << endl;

   cout << "CurrentTime(s)\tSendRate(Mb/s)\tRTT(ms)\tCWnd\tPktSndPeriod(us)\tRecvACK\tRecvNAK\tPktSndLoss\tPktSent\tPktRetrans" << endl;

  int64_t current_time = perf.msTimeStamp/1000;
  int64_t timeToRun=1000;

   while (current_time < timeToRun)
   {
      #ifndef WIN32
         sleep(1);
      #else
         Sleep(1000);
      #endif

      if (UDT::ERROR == UDT::perfmon(u, &perf))
      {
         cout << "perfmon: " << UDT::getlasterror().getErrorMessage() << endl;
         break;
      }

      cout << current_time << "\t\t"
           << perf.mbpsSendRate << "\t\t"
           << perf.msRTT << "\t" 
           << perf.pktCongestionWindow << "\t" 
           << perf.usPktSndPeriod << "\t\t\t" 
           << perf.pktRecvACK << "\t" 
           << perf.pktRecvNAK << "\t"
           << perf.pktSndLoss << "\t"
           << perf.pktSent << "\t"
           << perf.pktRetrans << endl;

      clientLog << current_time << "\t"
           << perf.mbpsSendRate << "\t"
           << perf.msRTT << "\t"
           << perf.pktCongestionWindow << "\t"
           << perf.usPktSndPeriod << "\t"
           << perf.pktRecvACK << "\t"
           << perf.pktRecvNAK << "\t"
           << perf.pktSndLoss << "\t"
           << perf.pktSent << "\t"
           << perf.pktRetrans << endl;

      current_time = perf.msTimeStamp/1000;
      // sendRateLog << "(" << current_time << ", " << perf.mbpsSendRate << "), ";
      // cwndLog << "(" << current_time << ", " << perf.pktCongestionWindow << "), ";
   }

   // sendRateLog << "]";
   // sendRateLog.close();
   // cwndLog << "]";
   // cwndLog.close();
   clientLog.close();

   cout << "monitor Done!" << endl;
   #ifndef WIN32
      return NULL;
   #else
      return 0;
   #endif
}
