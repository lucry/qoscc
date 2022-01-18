//
// Created by 葱葱 on 2021/10/11.
//

#define PY_SSIZE_T_CLEAN

#include <iostream>
#include "Python.h"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstring>
#include "../QoSCC/QoSCC.h"

using namespace std;

int main(int argc, char* argv[]){
    QoSCC QOS = QoSCC();
    int a = QOS.getSendingRate(1, 2, 3);

    return 0;
}