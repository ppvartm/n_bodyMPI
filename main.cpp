#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <mpi.h>

const int N = 2001;
int M = 0;
double time_interval = 20;
double h = time_interval / (N - 1);
double G = 6.67 / 100000000000;

using Mass = double;
struct Point_3d {
	Point_3d(double x = 0., double y = 0., double z = 0.) :x_(x), y_(y), z_(z) {}
	double x_;
	double y_;
	double z_;
};
struct Body {
	Point_3d r_;
	Point_3d v_;
};
struct K {
	Point_3d r_;
	Point_3d v_;
};

using Coordinates_and_speed_of_one_body = std::vector<Body>; //Координаты и скорость всех тел в текущий момент времени
using Masses_of_all_bodies = std::vector<Mass>; //массы всех тел

Point_3d operator*(const Point_3d& point, double a) {
	return { point.x_ * a, point.y_ * a , point.z_ * a };
}
Point_3d operator-(const Point_3d& point1, const Point_3d& point2) {
	return { point1.x_ - point2.x_, point1.y_ - point2.y_ , point1.z_ - point2.z_ };
}
Point_3d operator+(const Point_3d& point1, const Point_3d& point2) {
	return { point1.x_ + point2.x_, point1.y_ + point2.y_ , point1.z_ + point2.z_ };
}
double abs(const Point_3d& point1, const Point_3d& point2) {
	return sqrt((point1.x_ - point2.x_) * (point1.x_ - point2.x_)
		+ (point1.y_ - point2.y_) * (point1.y_ - point2.y_)
		+ (point1.z_ - point2.z_) * (point1.z_ - point2.z_));
}
double pow3(double x) {
	return x * x * x;
}
void Init(const std::string& file_name, Masses_of_all_bodies& m, Coordinates_and_speed_of_one_body& body) {
	std::ifstream file;
	file.open(file_name);
	file >> M;
	m.resize(M);
	body.resize(M);

	for (int i = 0; i < M; ++i) {
		file >> m[i];
		file >> body[i].r_.x_ >> body[i].r_.y_ >> body[i].r_.z_;
		file >> body[i].v_.x_ >> body[i].v_.y_ >> body[i].v_.z_;
	}
	file.close();
}

Point_3d r_foo(int j, const Coordinates_and_speed_of_one_body& body) {
	return body[j].v_;
}
Point_3d r_foo(int j, const Coordinates_and_speed_of_one_body& body, const std::vector<K>& k_) {
	return body[j].v_ + k_[j].v_ * 0.5;
}
Point_3d v_foo(int j, const Coordinates_and_speed_of_one_body& body, const Masses_of_all_bodies& m) {
	Point_3d result;
	for (int k = 0; k < M; ++k) {
		if (k == j)
			continue;
		result = result + (body[j].r_ - body[k].r_) * ((-G) * m[k] / pow3(abs(body[j].r_, body[k].r_)));
	}
	return result;
}
Point_3d v_foo(int j, const Coordinates_and_speed_of_one_body& body, const Masses_of_all_bodies& m, const std::vector<K>& k_) {
	Point_3d result;
	for (int k = 0; k < M; ++k) {
		if (k == j)
			continue;
		result = result + ((body[j].r_ + k_[j].r_ * 0.5) - (body[k].r_ + k_[k].r_ * 0.5)) *
			((-G) * m[k] / pow3(abs((body[j].r_ + k_[j].r_ * 0.5), (body[k].r_ + k_[k].r_ * 0.5))));
	}
	return result;
}

void ClearFiles() {
	for (int i = 0; i < M; ++i) {
		std::ofstream out;
		out.open("traj" + std::to_string(i) + ".txt");
		out.clear();
		out.close();
	}
}
void Write(const Coordinates_and_speed_of_one_body& body, int begin,int end) {
	for (int i = begin; i < end; ++i) {
		std::ofstream out;
		out.open("traj" + std::to_string(i) + ".txt", std::ios::app);
		out << body[i].r_.x_ << " " << body[i].r_.y_ << " " << body[i].r_.z_ << "\n";
		out.close();
	}
}


double CheckСonvergence(Coordinates_and_speed_of_one_body& ex_sol, Coordinates_and_speed_of_one_body& num_sol) {
	const int Q = 201;
	double er = 0;
	for (int i = 0; i < Q; ++i)
		er += sqrt((ex_sol[i].r_.x_ - num_sol[i].r_.x_) * (ex_sol[i].r_.x_ - num_sol[i].r_.x_) +
			(ex_sol[i].r_.y_ - num_sol[i].r_.y_) * (ex_sol[i].r_.y_ - num_sol[i].r_.y_) +
			(ex_sol[i].r_.z_ - num_sol[i].r_.z_) * (ex_sol[i].r_.z_ - num_sol[i].r_.z_));
	return er;
}

double RungeKuttaSecondOrder(Masses_of_all_bodies& m, Coordinates_and_speed_of_one_body& body,
	                       int this_process_id, const std::vector<int>& displaces, const std::vector<int>& sizes_of_data_for_each_process,
	                       MPI_Datatype* MPI_BODY) {
	std::vector<K> k1(M);
	std::vector<K> k2(M);
    
	auto last(body);

	ClearFiles();
	//begin-end - участок обработки для текущего потока 
	auto begin = displaces[this_process_id];
	auto end = displaces[this_process_id] + sizes_of_data_for_each_process[this_process_id];

	auto t1 = MPI_Wtime();

	for (int i = 1; i < N; ++i) {  //N - количество шагов по времени
		for (int j = begin; j < end; ++j) {
			k1[j].r_ = r_foo(j, last) * h;
			k1[j].v_ = v_foo(j, last, m) * h;
		}
		MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, k1.data(), sizes_of_data_for_each_process.data(), displaces.data(), *MPI_BODY, MPI_COMM_WORLD);
		for (int j = begin; j < end; ++j) {
			k2[j].r_ = r_foo(j, last, k1) * h;
			k2[j].v_ = v_foo(j, last, m, k1) * h;
			body[j].r_ = last[j].r_ + k2[j].r_;
			body[j].v_ = last[j].v_ + k2[j].v_;
		}
		MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, body.data(), sizes_of_data_for_each_process.data(), displaces.data(), *MPI_BODY, MPI_COMM_WORLD);
	    Write(body, begin, end);
		body.swap(last);
	}
	return MPI_Wtime() - t1;
}

double Solve() {

	int count_of_process;
	int this_process_id;
	MPI_Comm_size(MPI_COMM_WORLD, &count_of_process);
	MPI_Comm_rank(MPI_COMM_WORLD, &this_process_id);

	MPI_Datatype MPI_BODY;
	const int count_of_fields = 2;
	std::vector<int> len_of_each_fields = { 3, 3 };
	std::vector<MPI_Aint> displace = { offsetof(K, r_), offsetof(K, v_) };
	std::vector<MPI_Datatype> type_of_fields = { MPI_DOUBLE, MPI_DOUBLE };
	MPI_Type_create_struct(count_of_fields, len_of_each_fields.data(), displace.data(), type_of_fields.data(), &MPI_BODY);
	MPI_Type_commit(&MPI_BODY);

	Masses_of_all_bodies m;
	Coordinates_and_speed_of_one_body body;

	if (this_process_id == 0)
		Init("init.txt", m, body);
	MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (this_process_id != 0) {
		m.resize(M);
		body.resize(M);
	}

	MPI_Bcast(m.data(), M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(body.data(), M, MPI_BODY, 0, MPI_COMM_WORLD);

	std::vector<int> displaces(count_of_process);
	std::vector<int> sizes_of_data_for_each_process(count_of_process);

	if (this_process_id == 0) {
		for (int i = 0; i < count_of_process - 1; ++i) {
			sizes_of_data_for_each_process[i] = M / count_of_process;
			displaces[i + 1] = displaces[i] + sizes_of_data_for_each_process[i];
		}
		sizes_of_data_for_each_process[count_of_process - 1] = M / count_of_process + M % count_of_process;
	}

	MPI_Bcast(displaces.data(), count_of_process, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(sizes_of_data_for_each_process.data(), count_of_process, MPI_INT, 0, MPI_COMM_WORLD);

	return RungeKuttaSecondOrder(m, body, this_process_id, displaces, sizes_of_data_for_each_process, &MPI_BODY);
	
}
void GenerateFile(int sz) {
	std::ofstream init_file;
	init_file.open("init.txt");
	init_file << sz << '\n';

	auto random_coordinate = []() {
		double a = -100;
		double b = 100;
		return (rand() / (double)RAND_MAX) * (b - a) + a;
	};
	auto random_speed = []() {
		double a = -10;
		double b = 10;
		return (rand() / (double)RAND_MAX) * (b - a) + a;
    };
	auto random_mass = []() {
		double small = 88103241.227;
		double medium = 8810324116.227;
		double large = 881032411600.227;
		int id = 1 + (rand() % 3);
		if (id == 1) return small;
		if (id == 2) return medium;
		return large;
		};
	for (int i = 0; i < sz; ++i) {
		init_file << random_mass() << "  " << random_coordinate() << " " << random_coordinate() << " " << random_coordinate() << " ";
		init_file << random_speed() << " " << random_speed() << " " << random_speed() << "\n";
	}
	init_file.close();
}

int main() {
	MPI_Init(NULL, NULL);

	int count_of_process;
	int this_process_id;
	MPI_Comm_size(MPI_COMM_WORLD, &count_of_process);
	MPI_Comm_rank(MPI_COMM_WORLD, &this_process_id);
	auto result_time = Solve();
	if (this_process_id == 0)
		std::cout <<  result_time << "\n";
	//GenerateFile(20000);
	
	MPI_Finalize();
}