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
#include "cc.h"
#include "test_util.h"
#include "../QoSCC/QoSCC.h"
#include <sys/stat.h>
#include <time.h>
//#include "jsoncpp/json/json.h" // https://github.com/open-source-parsers/jsoncpp

using namespace std;

#ifndef WIN32
void* monitor(void*);
#else
DWORD WINAPI monitor(LPVOID);
#endif

#ifndef WIN32
void* recvdata(void*);
#else
DWORD WINAPI recvdata(LPVOID);
#endif

int client_seq=0;
int64_t timeToRun=1000;// 默认运行60秒
char percentLoss[3];
void Create_Folders(const char* dir);

int main(int argc, char* argv[])
{
   if ( 1 != argc && 3 != argc && 5 != argc && 7!=argc) {
      cout << "usage:" << argv[0] <<" -p(optional) [server_port] -t(optional) [time_to_run]" << endl;
      return 0;
   }

   char opt;
   char optarg_[10];
   string service("9000");
   while( (opt = getopt(argc, argv, "t:p:l:")) != -1 ){
      switch (opt)
      {
         case 't':
            timeToRun = atoi(optarg);
            if(timeToRun==0){
               cout << "Please input correct value after -t" << endl;
               return -1;
            }
            break;
         case 'p':
            service = optarg;
            break;
         case 'l':
            strcpy(percentLoss, optarg);
            break;
         default:
            cout << "usage:" << argv[0] <<" -p(optional) [server_port] -t(optional) [time_to_run]" << endl;
            return 0;
            break;
      }
   }


   // Automatically start up and clean up UDT module.
   UDTUpDown _udt_;

   addrinfo hints;
   addrinfo* res;

   memset(&hints, 0, sizeof(struct addrinfo));

   hints.ai_flags = AI_PASSIVE;
   hints.ai_family = AF_INET;
   hints.ai_socktype = SOCK_STREAM;
   //hints.ai_socktype = SOCK_DGRAM;

   if (0 != getaddrinfo(NULL, service.c_str(), &hints, &res))
   {
      cout << "illegal port number or port is busy.\n" << endl;
      return 0;
   }

   UDTSOCKET serv = UDT::socket(res->ai_family, res->ai_socktype, res->ai_protocol);

   // UDT Options
//   UDT::setsockopt(serv, 0, UDT_CC, new CCCFactory<cubic>, sizeof(CCCFactory<cubic>));
//    UDT::setsockopt(serv, 0, UDT_CC, new CCCFactory<QoSCC>, sizeof(CCCFactory<QoSCC>));
   //UDT::setsockopt(serv, 0, UDT_MSS, new int(9000), sizeof(int));
   //UDT::setsockopt(serv, 0, UDT_RCVBUF, new int(10000000), sizeof(int));
   //UDT::setsockopt(serv, 0, UDP_RCVBUF, new int(10000000), sizeof(int));

   if (UDT::ERROR == UDT::bind(serv, res->ai_addr, res->ai_addrlen))
   {
      cout << "bind: " << UDT::getlasterror().getErrorMessage() << endl;
      return 0;
   }

   freeaddrinfo(res);

   cout << "server is ready at port: " << service << endl;

   if (UDT::ERROR == UDT::listen(serv, 10))
   {
      cout << "listen: " << UDT::getlasterror().getErrorMessage() << endl;
      return 0;
   }

   sockaddr_storage clientaddr;
   int addrlen = sizeof(clientaddr);

   UDTSOCKET recver;

   while (true)
   {
      if (UDT::INVALID_SOCK == (recver = UDT::accept(serv, (sockaddr*)&clientaddr, &addrlen)))
      {
         cout << "accept: " << UDT::getlasterror().getErrorMessage() << endl;
         return 0;
      }

      client_seq++;

      char clienthost[NI_MAXHOST];
      char clientservice[NI_MAXSERV];
      getnameinfo((sockaddr *)&clientaddr, addrlen, clienthost, sizeof(clienthost), clientservice, sizeof(clientservice), NI_NUMERICHOST|NI_NUMERICSERV);
      cout << "new connection: " << clienthost << ":" << clientservice << endl;

      #ifndef WIN32
         pthread_t rcvthread;
         pthread_create(&rcvthread, NULL, recvdata, new UDTSOCKET(recver));
         pthread_detach(rcvthread);
      #else
         CreateThread(NULL, 0, recvdata, new UDTSOCKET(recver), 0, NULL);
      #endif
   }

   UDT::close(serv);

   return 0;
}

#ifndef WIN32
void* recvdata(void* usocket)
#else
DWORD WINAPI recvdata(LPVOID usocket)
#endif
{
   UDTSOCKET recver = *(UDTSOCKET*)usocket;
   delete (UDTSOCKET*)usocket;
   pthread_create(new pthread_t, NULL, monitor, &recver);// 在启动接收线程后启动监控线程

   char* data;
   int size = 100000;
   data = new char[size];

   while (true)
   {
      int rsize = 0;
      int rs;
      while (rsize < size)
      {
         int rcv_size;
         int var_size = sizeof(int);
         UDT::getsockopt(recver, 0, UDT_RCVDATA, &rcv_size, &var_size);
         if (UDT::ERROR == (rs = UDT::recv(recver, data + rsize, size - rsize, 0)))
         {
            cout << "recv:" << UDT::getlasterror().getErrorMessage() << endl;
            break;
         }

         rsize += rs;
      }

      if (rsize < size)
         break;
   }

   delete [] data;

   UDT::close(recver);

   #ifndef WIN32
      return NULL;
   #else
      return 0;
   #endif
}

#ifndef WIN32
void* monitor(void* s)
#else
DWORD WINAPI monitor(LPVOID s)
#endif
{
  UDTSOCKET u = *(UDTSOCKET*)s;
  int i = 0;

  UDT::TRACEINFO perf;

  int64_t current_time = perf.msTimeStamp/1000;
  time_t seconds_since_1970 = time(NULL);

  ofstream recvRateLog;
  char file_prefix[]="/home/QoSCC/plot/server-log-";
  char file_name1[100];
  sprintf(file_name1,"%s%d%s",file_prefix,client_seq,".txt");//prefix是前缀，将/home/tie/vemu_project/c++/verus-UDTP-master/src/plot/statistics/recvRate-log-client_seq.txt存入file_name
  remove(file_name1);//删除这个文件，以便下面再创建
  recvRateLog.open(file_name1, ios::out | ios::app);//打开文件，app表示没有则创建，out表示从内存写入文件
//  recvRateLog << "[";


  /*// jsoncpp - 用于记录UDT数据至日志文件
  Json::Value jsonLog = Json::objectValue;
  jsonLog["time"] = Json::arrayValue;
  jsonLog["recvRate"] = Json::arrayValue;
  jsonLog["totalRate"] = Json::arrayValue;*/

  cout << "Time\tRecv Rate(Mb/s)\t\tTotal Rate(Mb/s)\tPackets Recvd\tPktRcvLoss" << endl;
  recvRateLog << "Time\tRecv Rate(Mb/s)\t\tTotal Rate(Mb/s)\tPackets Recvd\tPktRcvLoss" << endl;

  double totalRecv=0;
  double mbpsTotalRate=0;

  while (current_time < timeToRun) {
    ++i;
#ifndef WIN32
    sleep(1);
#else
    Sleep(1000);
#endif

    if (UDT::ERROR == UDT::perfmon(u, &perf)) {
      cout << "perfmon: " << UDT::getlasterror().getErrorMessage() << endl;
      break;
    }
    current_time = perf.msTimeStamp/1000; //s
    totalRecv += perf.mbpsRecvRate;
    mbpsTotalRate = totalRecv / current_time;

    cout << current_time << "\t\t" << perf.mbpsRecvRate << "\t\t" << mbpsTotalRate
      << "\t\t\t" << perf.pktRecv << "\t"
           << perf.pktRcvLoss << endl;

//    recvRateLog<<"("<<current_time<<","<<perf.mbpsRecvRate<<"),";
    recvRateLog << current_time << "\t" << perf.mbpsRecvRate << "\t" << mbpsTotalRate
      << "\t" << perf.pktRecv << "\t"
           << perf.pktRcvLoss << endl;
    /*Json::Value::Int64  current_time_ = current_time;
    jsonLog["time"].append(current_time_);
    jsonLog["recvRate"].append(perf.mbpsRecvRate);
    jsonLog["totalRate"].append(mbpsTotalRate);*/
  }
//  recvRateLog<<"("<<current_time<<","<<0<<"),]";
    recvRateLog.close();
    UDT::close(u);
    cerr << "close, Done!" << endl;
  
  /* 默认使用这个
  char file_path[] = "./statistics";
  
  // 判断filepath文件夹是否存在，若不存在则创建（否则将无法创建出文件）
  if( access(file_path, F_OK)==-1 ){// 文件夹不存在, https://www.cnblogs.com/whwywzhj/p/7801409.html
    mkdir(file_path, S_IRWXO|S_IRWXG|S_IRWXU);// https://www.jianshu.com/p/06a0da1f6389
  }
  */

  /* 实验一 */
  char file_path[100]="./statistics/cubicstat";
//   if(access(file_path, F_OK)==-1){ 
//     mkdir(file_path, S_IRWXO|S_IRWXG|S_IRWXU);
//   }
  sprintf(file_path, "%s%s", "./statistics/cubicstat/loss=", percentLoss);
//   if(access(file_path, F_OK)==-1){ 
//     mkdir(file_path, S_IRWXO|S_IRWXG|S_IRWXU);
//   }

  // 判断filepath文件夹是否存在，若不存在则创建（否则将无法创建出文件）
  char path_for_judge[50];
  for(int i=1; ; i++){
    strcpy(path_for_judge, file_path);
    sprintf(path_for_judge, "%s%s%d", file_path, "/", i);
    if(access(path_for_judge, F_OK)==-1){ 
       strcpy(file_path, path_for_judge);
       //mkdir(file_path, S_IRWXO|S_IRWXG|S_IRWXU);
       Create_Folders(file_path);
       break;
     } 
  }

  // 根据时间为文件命名 https://blog.csdn.net/weixin_43977523/article/details/88868651
  char file_name[200];
  time_t file_time;// file time
  time(&file_time);
  struct tm* p = localtime(&file_time);
  sprintf(file_name, "%s%s%04d%02d%02d%02d%02d%02d%s", file_path, "/recvLog-", 1900+p->tm_year, p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec, ".json");
  while( access(file_name, F_OK)==0 ){
    strcat(file_name, "_"); // TODO:（还未测试是否可用）为应对同一秒结束导致重名的情况，若文件已存在则在文件名后加一个下划线
  }

  ofstream recvLog;
  recvLog.open(file_name, ios::out | ios::app);
  /*recvLog << Json::FastWriter().write(jsonLog) << endl;*/

  cerr << "recv_monitor close, Done!" << endl;
  cerr << "recvLog name: " << file_name << endl;
  UDT::close(u);

#ifndef WIN32
  return NULL;
#else
  return 0;
#endif
}

void Create_Folders(const char* dir){
  char order[100] = "mkdir -p ";
  strcat(order, dir); 
  system(order);
} 
