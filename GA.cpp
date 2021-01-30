#include <iostream>
#include <algorithm>
#include <cstring>
#include <vector>
#include <string>
#include <queue>
#include <time.h>
#include <cmath>
#include <map>
#include <stdlib.h>
#include <cstdlib>
#include <conio.h>
#include <omp.h>

using namespace std;
#define NUM_THREAD 5
const int N = 200, NICH = 10, POPUL = 50;
int iteration = 1000;
int n, nich_num = 5, popul = 20, L = 3, day = 5;
double dist_line = 1, hour_per_km = 0.05, trip_time = 8, start_time = 9, upper_cost = 900, lower_cost = 500;
double transit = 3.87, rating = 3.74, diversity = 2.39, punish = 5, cost_up = 0.2, cost_lp = 0.1;
double cr = 0.6, mr = 0.01;
double type_list[51] = {13, 3, 2, 2, 1, 1, 1, 2, 3, 1, 6, 1, 9, 4, 1, 1, 7, 1, 2, 3, 3, 4, 6, 1, 2, 1, 36, 2, 3, 2, 4, 1, 2, 3, 1, 1, 4, 2, 1, 1, 34, 1, 1, 1, 1, 2, 5, 3, 1, 1, 3};
double type_sum = 197;
double eps = 1e-5;
double tJ = 0, tN = 0, tM = 0, tR = 0, tA = 0, tRR = 0, tC = 0;
double nJ = 0, nN = 0, nM = 0, nR = 0, nA = 0, nRR = 0, nC = 0;
struct POI
{
    char type[60];
    double longitude, latitude;
    double trip_time, score;
    double open, close, price;
};// poi[N];
POI* poi = (POI*)malloc(N * sizeof(POI));
bool cmp_score(POI a, POI b){return a.score > b.score;}
bool cmp_index_score(int a, int b){return poi[a].score > poi[b].score;}
bool cmp_trip(POI a, POI b) {return a.trip_time == b.trip_time? a.score < b.score: a.trip_time > b.trip_time;}
struct cmp_index_trip
{
    bool operator() (int a, int b){return poi[a].trip_time == poi[b].trip_time? poi[a].score > poi[b].score: poi[a].trip_time < poi[b].trip_time;}
};


double Distance(int u, int v)
{
    if(u == 0 || v == 0 || u == v)
        return 0;
    double xu = poi[u].longitude;
    double yu = 90 - poi[u].latitude;
    double xv = poi[v].longitude;
    double yv = 90 - poi[v].latitude;
    return dist_line * 6371.004 * acos(sin(yu) * sin(yv) * cos(xu - xv) + cos(yu) * cos(yv)) * acos(-1) / 180.0;
}
//double dist[N][N];
double* dist = (double*)malloc(N * N * sizeof(double));
bool sort_dist(double A, double B)
{
    int u = A/(N + 1);
    int a = (int)A%(N + 1);
    int b = (int)B%(N + 1);
    return Distance(u, a) < Distance(u, b);
}
struct EDGE
{
    int u, v;
	EDGE(int _u, int _v)
	{
		u = _u; v = _v;
	}
    bool operator < (const EDGE &e) const
    {
        if(u == e.u)
            return v < e.v;
        return u < e.u;
    }
    bool operator == (const EDGE &e) const
    {
        return u == e.u && v == e.v;
    }
};
struct INDIV
{
	int length;
	int gene[50];
	//int* gene = (int*)malloc(50 * sizeof(int));
	//vector<EDGE> edge;  //����߼�����������ƶȼ���
	EDGE edge[50];
	//EDGE* edge = (EDGE*)malloc(50 * sizeof(EDGE));
	void makeEDGE()
	{
		int I = 0;
		//edge.clear();
		EDGE &x = *this->edge;
		//memset(edge, 0, 50 * sizeof(EDGE));
		for (int i = 1; i < day * (L + 1); i++)
		{
			if (gene[i - 1] == gene[i])
				continue;
			//edge.push_back(EDGE(gene[i-1], gene[i]));
			edge[I++] = EDGE(gene[i - 1], gene[i]);
		}
		//sort(edge.begin(), edge.end());
		sort(edge, edge + I);
		length = I;
	}
	void out()
	{
		for (int i = 0; i < day; i++)
		{
			for (int j = (day) * (L + 1) + 1; j < (day + 1) * (L + 1); j++)
			{
				cout << gene[j] << " ";
			}
			printf("| ");
		}
		printf("\n");
	}
};//total[N];
INDIV* total = (INDIV*)malloc(N * sizeof(INDIV));
double SI(INDIV u, INDIV v)
{
    int i = 0, j = 0; double Union = 0, Cross = 0;
    int I = u.length, J = v.length;
    while(i < I)
    {
        while(j < J && v.edge[j] < u.edge[i])
        {
            Union += 1;
            j++;
        }
        if(j < J && u.edge[i] == v.edge[j])
            Cross += 1;
        Union += 1;
        i++;
    }
    if(Union < 1)
        return 0;
    return Cross/Union;
}

//int nich[NICH][POPUL];
int* nich = (int*)malloc(NICH * POPUL * sizeof(int));
void read()
{
	FILE *file;
    file = freopen("NGAin.txt", "r", stdin);
    scanf("%d", &n);
    for(int i = 1; i <= n; i++){
        cin >> poi[i].longitude >> poi[i].latitude >> poi[i].trip_time >> poi[i].type
            >> poi[i].score >> poi[i].open >> poi[i].close >> poi[i].price;
        //cout << poi[i].longitude << poi[i].latitude << poi[i].trip_time
            //<< poi[i].score << poi[i].open << poi[i].close << poi[i].price <<endl;;
    }
	fclose(file);
}

void out(INDIV indiv)
{
    for(int i = 0; i <= day*(L + 1); i++)
        cout<<indiv.gene[i]<<" ";
	cout << endl;
}

double Judgement(INDIV indiv)
{
    double Transit = 0, Score = 0, Occupied = 0, Type = 0, Cost = 0;
    double number = 0;
    //double Type_count[55];
    double* Type_count = (double*)malloc(55 * sizeof(double));
    memset(Type_count, 0, 55 * sizeof(double));
    for(int i = 0; i < day*(L + 1) - 1; i++)
    {
        if(i%(L + 1) == 1 && indiv.gene[i] == 0)
            return -1e5;
        Transit += hour_per_km * Distance(indiv.gene[i], indiv.gene[i+1]);
        if(indiv.gene[i] != 0)
        {
            number += 1;
            Score += poi[indiv.gene[i]].score;
            Occupied += poi[indiv.gene[i]].trip_time;
            Cost += poi[indiv.gene[i]].price;
            string type = poi[indiv.gene[i]].type;
            for(int j = 0; j < type.size(); j++)
            {
                if(type[j] == '1')
                    Type_count[j] = 1;
            }
        }
    }
    if(number < day)
        return -1e5;
    for(int i = 0; i < 51; i++)
    {
        //Type += (1 - type_list[i]/type_sum)*Type_count[i] - (Type_count[i] == 0? type_list[i]: 0);
        Type += Type_count[i];
    }
    double f1 = -1/(double)day*Transit;
    double f2 = 1.0 /(double)number*Score;
    //double f3 = 0.05/(double)(number /* - 1*/)*Type;
    double f3 = Type / double(day);
    double f4;
    if(Cost > upper_cost)
    {
        f4 = (Cost - upper_cost) / number;
    }
    else{
        f4 = 0;
    }
    free(Type_count);
//    #pragma omp critical
//    cout << f1 << " " << f2 << " " << f3 << " " << (day * trip_time - Occupied - Transit) * (transit + (rating + diversity) / trip_time) / 10 / day << " " << f4 << endl;
    return transit*f1 + rating*f2 + diversity*f3 - (day * trip_time - Occupied - Transit) * (transit + (rating + diversity) / trip_time) / 10 / day - f4; //�������������δ����
}

bool Total_Time_Constrain(INDIV indiv, int d)
{
    double time = 0;
    for(int j = (d)*(L + 1); j < (d + 1)*(L + 1); j++) //�����ʱ������ʱ�� + ·��ʱ�䣬�����ǾƵ�
    {//cout<< j << " " << indiv.gene[j]<<endl;
        time += poi[indiv.gene[j]].trip_time + hour_per_km*Distance(indiv.gene[j], indiv.gene[j+1]);
    }
    return time <= trip_time;
}

bool Open_Time_Constrain(INDIV indiv, int d)
{
    double time = start_time;
    for(int j = (d)*(L + 1) + 1; indiv.gene[j] != 0; j++)
    {
        time = time + hour_per_km*Distance(indiv.gene[j-1], indiv.gene[j]);
        POI CUR = poi[indiv.gene[j]];
        time = max(time, CUR.open);
        if(time + CUR.trip_time > CUR.close)
            return false;

    }

    return true;
}

INDIV Repair(INDIV in)  //O((DL)n)
{
    for(int i = 0; i < day; i++)  //һ���촦��
    {
        for(int j = i*(L + 1) + 2; j < (i + 1)*(L + 1); j++)
        {
            int jj = j;
            while(jj-1 != i*(L + 1) && in.gene[jj] != 0 && in.gene[jj-1] == 0)
            {
                swap(in.gene[jj], in.gene[jj-1]);
                jj--;
            }
        }
        priority_queue<int, vector<int>, cmp_index_trip> q; //̰������ɾ����ʱ����
        map<int, int> ma;
        for(int j = i*(L + 1) + 1; in.gene[j] != 0; j++)
        {
            q.push(in.gene[j]);
            ma[in.gene[j]] = j;
        }//out(in);
        while(!Total_Time_Constrain(in, i))  //����һ�������ʱ���Լ��
        {
            int pos = ma[q.top()];
            q.pop();
            in.gene[pos] = 0;
            //cout<<"Repair\n";
            while(in.gene[++pos] != 0)
                swap(in.gene[pos-1], in.gene[pos]);

        }
        int cur = i*(L + 1) + 1;
        double time = start_time;
        while(in.gene[cur] != 0)  //����ÿ��������Ӫҵʱ��Լ��
        {
            time = time + hour_per_km*Distance(in.gene[cur-1], in.gene[cur]);
            POI CUR = poi[in.gene[cur]];
            time = max(time, CUR.open);
            if(time + CUR.trip_time > CUR.close)
            {
                in.gene[cur] = 0;
                int tmpcur = cur;
                while(in.gene[++tmpcur] != 0)
                    swap(in.gene[tmpcur-1],in.gene[tmpcur]);
            }
            else
                cur++;
        }
    }
    return in;
}

INDIV Augmentation(INDIV in)
{
    //bool vis[N];
	bool* vis = (bool*)malloc(N * sizeof(bool));
    memset(vis, 0, N * sizeof(bool));
    for(int i = 0; i < day*(L + 1); i++)
        vis[in.gene[i]] = 1;
    for(int i = 0; i < day; i++)
    {
        int occupied = 0;
        for(int j = i*(L + 1) + 1; j < (i + 1)*(L + 1); j++)
            if(in.gene[j] > 0)
                occupied++;
        if(occupied >= L)  //û�пռ��poi
            continue;
        INDIV res = in;
        int join = 0;
        double J = Judgement(res);
        for(int j = i*(L + 1) + 1; j < (i + 1)*(L + 1); j++)
        {
            int u = in.gene[j];
            if(u == 0)
                continue;
            double random = double(rand()*107%100 + 1)/100.0*dist[u*N + 0];
            int v = 0;//cout<<"check1"<<endl;
            while(random > eps)  //���ȡ��
            {
                random -= Distance(u, dist[u*N + ++v]);
                //cout<<random<<endl;
            }

            int pos = v;
            while(v > 0 && vis[(int)round(dist[u*N + v])])
                v--;
            if(v < 1)
            {
                v = pos;
                while(vis[(int)round(dist[u*N + v])])
                    v++;
            }
            v = dist[u*N + v];  //POI to be insert
            INDIV tmp = in;
            pos = j; int vi = v;
            while(vi != 0)  //Left
            {
                swap(vi, tmp.gene[pos++]);
            }
            try
            {
                double JT = Judgement(tmp);
                if(Total_Time_Constrain(tmp, i) && Open_Time_Constrain(tmp, i) && JT > J)
                {
                    res = tmp;
                    join = v;
                    J = JT;
                }
            }catch(exception e)
            {
                //cout<<e.what()<<endl;
            }
            //cout<<"checkb"<<endl;
            tmp = in;
            pos = j + 1, vi = v;
            while(vi != 0)  //Right
            {
                swap(vi, tmp.gene[pos++]);
            }
            double JT = Judgement(tmp);
            if(Total_Time_Constrain(tmp, i) && Open_Time_Constrain(tmp, i) && JT > J)
            {
                res = tmp;
                join = v;
                J = JT;
            }
        }
        if(join > 0)
        {
            in = res;
            vis[join] = true;
        }
    }
    in.makeEDGE();
	free(vis);
    return in;
}

void Init()
{
    bool vis[N];
    #pragma omp parallel
    {
        int i = omp_get_thread_num() + 1;
    for( ; i <= nich_num*popul; i += NUM_THREAD)
    //for(int i = 1; i <= nich_num*popul; i++)
    {
        memset(vis, 0, sizeof(vis));
        for(int j = 0; j <= (L + 1)*day; j++)
        {
            if(j % (L + 1) == 0)  //һ���������0
                total[i].gene[j] = 0;
            else
            {
                int x = rand()*107%n + 1;  //ֻҪ���ظ�����
                while(vis[x])
                {
                    x = rand()*107%n + 1;
                }
                total[i].gene[j] = x;
                vis[x] = true;
            }
        }
		//out(total[i]);
        total[i] = Repair(total[i]);

        total[i] = Augmentation(total[i]);
        //cout<<i<<": "; out(total[i]);
        //cout<<endl;
    }
    }
}

bool cmp_SI(int a, int b)
{
    int leader = a/(N + 1);

    if(a <= 0 || b <= 0 || leader <= 0)
    {
        cout <<leader << " "<< a <<" "<<b<<endl;
        throw "---";
    }

    a %= (N + 1);
    b %= (N + 1);

    return SI(total[leader], total[a]) > SI(total[leader], total[b]);
}
struct cmp_index_si
{
    bool operator() (int a, int b){return cmp_SI(a, b);}
};
void Neighborhood()
{

    //bool vis[N];
	bool* vis = (bool*)malloc(N * sizeof(bool));
    memset(vis, 0, N * sizeof(bool));
	int candidate[N];
	//int* candidate = (int*)malloc(N * sizeof(int));
	memset(candidate, 0, sizeof(candidate));
	double best = -1e5; int leader = 0;
    for(int i = 1; i < nich_num; i++)  //todo: 最后一个nich不需要分配
    {
		clock_t st = clock();
        int m = 1;
		best = -1e5;
		leader = 0;
		//#pragma omp parallel
        //for(int j = omp_get_thread_num() + 1; j <= nich_num*popul; j += NUM_THREAD)
        for(int j = 1; j <= nich_num*popul; j++)
        {
            if(!vis[j])
            {
				double J = Judgement(total[j]);
                //#pragma omp critical
                //{
                    candidate[m++] = j;
                    if(J > best)
                    {
                        best = J;
                        leader = j;
                    }
                //}
            }
        }
		clock_t fi = clock();
		//printf("first: %lf\n", double(fi - st));
        for(int j = 1; j < m; j++)
            candidate[j] += leader*(N + 1);
		clock_t se = clock();
		priority_queue<int, vector<int>, cmp_index_si> q; //̰������ɾ����ʱ����
		//printf("second: %lf\n", double(se - fi));
        //sort(candidate + 1, candidate + m, cmp_SI);
        for(int j = 1; j < m; j++)
            q.push(candidate[j]);
		clock_t th = clock();
		//printf("third: %lf\n", double(th - se));
        for(int j = 1; j <= popul; j++)
        {
            int cur = q.top();
            nich[i* POPUL + j] = cur%(N + 1);
            vis[cur%(N + 1)] = 1;
        }
		clock_t fo = clock();
		//printf("fourth: %lf\n", double(fo - th));
    }
	best = -1e5;
	leader = 0;
	int j = 1;
	for (int i = 1; i <= nich_num * popul; i++)
	{
		if (!vis[i])
		{
		    double J = Judgement(total[i]);
			if (J > best)
			{
				best = J;
				leader = j;
			}
			nich[nich_num * POPUL + (j++)] = i;
		}
	}
	swap(nich[nich_num * POPUL + 1], nich[nich_num * POPUL + leader]);
	//free(candidate);
	free(vis);
}

INDIV FixedMutation(INDIV in, int pos)
{
    bool vis[N];
    memset(vis, 0, sizeof(vis));
    for(int i = 0; i < day*(L + 1); i++)
        vis[in.gene[i]] = 1;
    int new_ge = rand()*107%n + 1;
    while(vis[new_ge])
        new_ge = rand()*107%n + 1;
    in.gene[pos] = new_ge;
    return in;
}

void Crossover(INDIV p1, INDIV p2, INDIV &s1, INDIV &s2)
{
    int x = rand()*107%(day*(L + 1)) + 1;
    int y = rand()*107%(day*(L + 1)) + 1;
    if(x > y) swap(x, y);
    map<int, int> ma;
    s1 = p1; s2 = p2;
    //bool vis1[N], vis2[N];
	bool* vis1 = (bool*)malloc(N * sizeof(bool));
	bool* vis2 = (bool*)malloc(N * sizeof(bool));
    memset(vis1, 0, N * sizeof(bool));
    memset(vis2, 0, N * sizeof(bool));
    for(int i = x; i <= y; i++)
    {
		int xx = p1.gene[i], yy = p2.gene[i];
        if(xx > 0 && yy > 0)
        {
            //ma[p1.gene[i]] = p2.gene[i];

            if(ma[xx] != 0 && ma[yy] != 0)
            {
                ma[ma[xx]] = ma[yy];
                ma[ma[yy]] = ma[xx];
            }
            else if(ma[xx] != 0)
            {
                ma[ma[xx]] = yy;
                ma[yy] = ma[xx];
            }
            else if(ma[yy] != 0)
            {
                ma[ma[yy]] = xx;
                ma[xx] = ma[yy];
            }
            else
            {
                ma[xx] = yy;
                ma[yy] = xx;
            }
        }
        else if(xx > 0)
        {
			if (ma[xx] != 0)
				ma[ma[xx]] = -1;
			else
		        ma[xx] = -1;
        }
        else if(yy > 0)
        {
			if (ma[yy] != 0)
				ma[ma[yy]] = -1;
			else
				ma[yy] = -1;

        }
        swap(s1.gene[i], s2.gene[i]);
        if(s1.gene[i] > 0)
            vis1[s1.gene[i]] = 1;
        if(s2.gene[i] > 0)
            vis2[s2.gene[i]] = 1;
    }

    for(int i = 1; i < x; i++)
    {
        if(vis1[s1.gene[i]])
        {
            s1.gene[i] = max(0, ma[s1.gene[i]]);
        }
        if(vis2[s2.gene[i]])
        {
            s2.gene[i] = max(0, ma[s2.gene[i]]);
        }
    }
    for(int i = y+1; i < day*(L + 1); i++)
    {
        if(vis1[s1.gene[i]])
        {
            s1.gene[i] = max(0, ma[s1.gene[i]]);
        }
        if(vis2[s2.gene[i]])
        {
            s2.gene[i] = max(0, ma[s2.gene[i]]);
        }
    }

	free(vis1); free(vis2);
    //cout<<x<<" "<<y<<" ";out(s2);
}


INDIV Mutation(INDIV in)
{
    int pos = rand()*107%(day*(L + 1));
    if(pos % (L + 1) == 0)
        pos++;
    return FixedMutation(in, pos);
}

void Replacement(INDIV* son, int ns, int nich_id)
{
    //cout<<son.size()<<endl;
    for(int i = 0; i < ns; i++)
    {
//        #pragma omp critical
//            out(son[i]);
        INDIV cur = son[i];
        double near = -1e5; int best = 0;
        double S;
        for(int j = 1; j <= popul; j++)
        {
            S = SI(cur, total[nich[nich_id* POPUL + j]]);
            if(S > near)
            {
                near = S;
                best = j;
            }
        }
        if(Judgement(cur) > Judgement(total[nich[nich_id* POPUL + best]]))
        {
            total[nich[nich_id* POPUL + best]] = cur;
        }
    }
}

int main(int argc, char* argv[])
{
    transit = atof(argv[1]), rating = atof(argv[2]), diversity = atof(argv[3]), upper_cost = atof(argv[4]), lower_cost = atof(argv[5]), day = atof(argv[6]);
	//transit = 0.44123, rating = 0.17753, diversity = 0.38124, upper_cost = 7404, lower_cost = 6256;
	omp_set_num_threads(NUM_THREAD);
    read();
    clock_t SS = clock();
    srand((int)time(0));
    for(int i = 1; i <= n; i++)
    {
        dist[i*N + 0] = 0;
        for(int j = 1; j <= n; j++)
        {
            dist[i*N + j] = i*(N + 1) + j;
            dist[i*N + 0] += Distance(i, j);
        }
        sort(dist + i*N + 1, dist + i*N + 1 + n, sort_dist);
        for(int j = 1; j <= n; j++)
        {
            dist[i*N + j] -= i*(N + 1);
        }

    }
    Init();
    while(iteration--)
    {
        try
        {
            clock_t a = clock();
            Neighborhood();
            nN += 1;
            clock_t b = clock();
            tN += double(b - a)/CLOCKS_PER_SEC;
        }catch(exception e)
        {
            //cout<<e.what()<<endl;
        }
        //cout<<"Neighborhood success\n";
        #pragma omp parallel
        //for(int i = 1; i <= nich_num; i++)
        {
            //vector<INDIV> parent;
            int i = omp_get_thread_num() + 1;
			INDIV* parent = (INDIV*)malloc(POPUL * sizeof(INDIV));
			int np = 0;
			double cross;
            for(int j = 1; j <= popul; j++)  //Crossover
            {
                cross = rand()*107%100/100.0;
                if(cross <= cr)
                {
                    //parent.push_back(total[nich[i* NICH + j]]);
					parent[np++] = (total[nich[i * POPUL + j]]);
                }
            }
            //vector<INDIV> son;
			INDIV* son = (INDIV*)malloc(POPUL * sizeof(INDIV));
			int ns = 0;
            for(int j = 0; j <np; j+=2)
            {
                INDIV s1 = parent[j], s2 = parent[(j+1)%np];
                clock_t a = clock();
                Crossover(parent[j], parent[(j+1)%np], s1, s2);
                clock_t b = clock();
                nC += 1;
                tC += double(b - a)/CLOCKS_PER_SEC;
                //son.push_back(s1); son.push_back(s2);
				son[ns++] = s1; son[ns++] = s2;
            }
            //

            for(int j = 0; j < ns; j++)
            {
                clock_t a = clock();
                son[j] = Mutation(son[j]);
                clock_t b = clock();

                son[j] = Repair(son[j]);
                clock_t c = clock();

                son[j] = Augmentation(son[j]);
                clock_t d = clock();

                #pragma omp critical
                {
//                    out(son[j]);
                    tM += double(b - a)/CLOCKS_PER_SEC;
                    tR += double(c - b)/CLOCKS_PER_SEC;
                    tA += double(d - c)/CLOCKS_PER_SEC;
                    nM += 1; nR += 1; nA += 1;
                }

            }
            try
            {
                clock_t a = clock();
                Replacement(son, ns, i);
                clock_t b = clock();
                #pragma omp critical
                {
                    tRR += double(b - a)/CLOCKS_PER_SEC;
                    nRR += 1;
                }

            }catch(exception e)
            {
                //cout<<e.what()<<endl;
            }
            //cout<<son.size()<<endl;
			free(parent);
			free(son);
			//cout<<Judgement(total[nich[1* POPUL + 1]]) <<endl;
        }


    }
    clock_t EE = clock();
    //out(total[nich[1* POPUL + 1]]);
    cout<<Judgement(total[nich[1* POPUL + 1]]) <<endl;
    //printf("Tot: %f\n", double(EE - SS)/CLOCKS_PER_SEC);
    //printf("\nNei: %f %f\nMut: %f %f\nCro: %f %f\nRepair: %f %f\nAug: %f %f\nReplace: %f %f\n", tN, tN/nN, tM, tM/nM, tC, tC/nC, tR, tR/nR, tA, tA/nA, tRR, tRR/nRR);


    return 0;
}
