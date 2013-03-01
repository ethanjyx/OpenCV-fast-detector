#include<iostream>
#include<string>
#include<vector>
#include<fstream>
#include<sstream>
#include<stdio.h>
#include<stdlib.h>

using namespace std;

main(){
	ifstream ifs("myhog.out", ifstream::in);
	string instring;
	stringstream ss;
	vector<vector<double> > compuTime(8);
/*	vector<double> reDes;
	vector<double> Desfft;
	vector<double> reDet;
	vector<double> Defft;
	vector<double> multi;
	vector<double> inverseFFt;
	vector<double> overall;
*/	unsigned found, found2;
	double time = 0;
	int i = 0;
	while(1){
		getline(ifs, instring);
		if(instring.empty()) break;
		found = instring.find('=');
		found2 = instring.find("ms");
		instring = instring.substr(found + 2, found2 - found - 2);
//		ss << instring;
//		cout << ss.str() << endl;
//		cout << instring << endl;
//		ss >> time;
		time = atof(instring.c_str());
//		cout << time << endl;
		compuTime[i % 8].push_back(time);
		i++;
	}
	double sum = 0;;
	for(unsigned i = 0; i < 8; i++){
		sum = 0;
		for(unsigned j = 0; j < compuTime[i].size(); j++){
			sum += compuTime[i][j];
			//cout  << "!!!" << endl;
//			cout  << compuTime[i][j] << endl;
		}
		cout << (double) (sum / compuTime[i].size()) << endl;
	}

}	
