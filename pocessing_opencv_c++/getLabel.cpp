#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include "getLabel.h"

using namespace std;

char getLabel(ifstream& file1,string& id)
{
  
  int label;
  
  //char label;
  /***********************************sex criteria*****************************************/
  
    string line, line1, line2;
    getline(file1, line);
    istringstream iss1(line);
    string sex;
    iss1 >> id >> sex;
    // getline(iss1, line1, ':' );
    //cout << id << endl;
    //getline(file1, line);
    //istringstream iss2(line);
    //getline(iss2, line2, ':');   
    //iss2 >> sex;
    //cout << "sex ---- " << sex << endl;

     if (sex ==  "M") {     
       label=0;
    } else if(sex ==  "F"){
       label=1;
           }

    cout<< label << "============id========" <<id << endl;
  
/***********************************age criteria*****************************************/
 /*
    string line, line1, line2,line3;
    getline(file1, line);
    istringstream iss1(line);
    getline(iss1, line1, ':');
    //string id;
    iss1 >> id;
    getline(file1, line);
    getline(file1, line);
    istringstream iss4(line);
    getline(iss4, line2, ':');
    int year;
    iss4 >> year;
    int age;
    age = 2015 - year;

        if (age < 30) {
       label=0;
    } else {
       label=1;
    }
*/
    /***********************************ethinicity*****************************************/
/*
    
    string line, line1, line2,line3;
    getline(file1, line);
    istringstream iss1(line);
    getline(iss1, line1, ':');
    iss1 >> id;
    getline(file1, line1);
    getline(file1, line2);
    getline(file1, line3);
    istringstream iss4(line3);
    getline(iss4, line3, ':');
    string eth;
    iss4 >> eth;

        if (eth ==  "White") {
       label=0;
    } else if (eth ==  "Black" || eth ==  "Indian" || eth ==  "Hispanics"){ 
       label=1;
    } else if (eth ==  "Middle"){ 
       label=2;
    } else if (eth ==  "Asian(Chinese)") {    
       label=3;
    }

   */ 

/**************************CURTIN FACES database*****************************************/

  return label;
}

  
