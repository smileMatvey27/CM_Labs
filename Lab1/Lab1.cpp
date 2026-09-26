#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;
using Matrix = vector<vector<double>>;
using Vector = vector<double>;

const double EPS_PIVOT = 1e-14;

void clearInput() {
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
}

Matrix identity(int n) {
    Matrix E(n, Vector(n, 0.0));
    for (int i = 0; i < n; ++i) E[i][i] = 1.0;
    return E;
}

void printVector(const Vector& v, const string& name = "x") {
    cout << name << " = [\n";
    for (double x : v) cout << "  " << setprecision(12) << x << "\n";
    cout << "]\n";
}

void printMatrix(const Matrix& A, const string& name = "A") {
    cout << name << " =\n";
    for (const auto& row : A) {
        cout << "  ";
        for (double v : row) cout << setw(14) << setprecision(8) << v << " ";
        cout << "\n";
    }
}

Matrix readMatrix(int n, const string& name = "A") {
    cout << "Enter matrix " << name << " (" << n << " rows, " << n << " numbers each):\n";
    Matrix A(n, Vector(n));
    for (int i = 0; i < n; ++i) {
        while (true) {
            cout << name << "[" << i + 1 << "]: ";
            bool ok = true;
            for (int j = 0; j < n; ++j) {
                if (!(cin >> A[i][j])) { ok = false; break; }
            }
            if (ok) break;
            cout << "Need exactly " << n << " numbers.\n";
            cin.clear();
            clearInput();
        }
    }
    return A;
}

Vector readVector(int n, const string& name = "b") {
    while (true) {
        cout << "Enter vector " << name << " (" << n << " numbers): ";
        Vector v(n);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (!(cin >> v[i])) { ok = false; break; }
        }
        if (ok) return v;
        cout << "Need exactly " << n << " numbers.\n";
        cin.clear();
        clearInput();
    }
}

double diffNormInf(const Vector& a, const Vector& b) {
    double m = 0.0;
    size_t n = min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) m = max(m, fabs(a[i] - b[i]));
    return m;
}

double matNormInf(const Matrix& A) {
    double m = 0.0;
    for (const auto& row : A) {
        double s = 0.0;
        for (double v : row) s += fabs(v);
        m = max(m, s);
    }
    return m;
}

void luDecomposition(Matrix A, Matrix& L, Matrix& U, Vector& perm, int& swaps) {
    int n = (int)A.size();
    U = A;
    L = identity(n);
    perm.resize(n);
    for (int i = 0; i < n; ++i) perm[i] = i;
    swaps = 0;

    for (int k = 0; k < n - 1; ++k) {
        int pivot = k;
        double maxAbs = fabs(U[k][k]);
        for (int i = k + 1; i < n; ++i)
            if (fabs(U[i][k]) > maxAbs) { maxAbs = fabs(U[i][k]); pivot = i; }

        if (maxAbs < EPS_PIVOT) throw runtime_error("Matrix is singular");

        if (pivot != k) {
            swap(U[k], U[pivot]);
            swap(perm[k], perm[pivot]);
            for (int j = 0; j < k; ++j) swap(L[k][j], L[pivot][j]);
            swaps++;
        }

        for (int i = k + 1; i < n; ++i) {
            L[i][k] = U[i][k] / U[k][k];
            for (int j = k; j < n; ++j) U[i][j] -= L[i][k] * U[k][j];
            U[i][k] = 0.0;
        }
    }
    if (fabs(U[n - 1][n - 1]) < EPS_PIVOT)
        throw runtime_error("Matrix is singular");
}

Vector forwardSub(const Matrix& L, const Vector& b) {
    int n = (int)L.size();
    Vector y(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < i; ++j) s += L[i][j] * y[j];
        y[i] = (b[i] - s) / L[i][i];
    }
    return y;
}

Vector backSub(const Matrix& U, const Vector& y) {
    int n = (int)U.size();
    Vector x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double s = 0.0;
        for (int j = i + 1; j < n; ++j) s += U[i][j] * x[j];
        if (fabs(U[i][i]) < EPS_PIVOT) throw runtime_error("Zero diagonal in U");
        x[i] = (y[i] - s) / U[i][i];
    }
    return x;
}

Vector luSolve(const Matrix& A, const Vector& b) {
    Matrix L, U; Vector perm; int swaps;
    luDecomposition(A, L, U, perm, swaps);
    Vector pb(b.size());
    for (size_t i = 0; i < b.size(); ++i) pb[i] = b[perm[i]];
    return backSub(U, forwardSub(L, pb));
}

double determinantFromLU(const Matrix& A) {
    Matrix L, U; Vector perm; int swaps;
    luDecomposition(A, L, U, perm, swaps);
    double det = (swaps % 2 == 0) ? 1.0 : -1.0;
    for (int i = 0; i < (int)U.size(); ++i) det *= U[i][i];
    return det;
}

Matrix inverseFromLU(const Matrix& A) {
    int n = (int)A.size();
    Matrix L, U; Vector perm; int swaps;
    luDecomposition(A, L, U, perm, swaps);
    Matrix inv(n, Vector(n, 0.0));
    for (int col = 0; col < n; ++col) {
        Vector e(n, 0.0); e[col] = 1.0;
        Vector pe(n);
        for (int i = 0; i < n; ++i) pe[i] = e[perm[i]];
        Vector x = backSub(U, forwardSub(L, pe));
        for (int i = 0; i < n; ++i) inv[i][col] = x[i];
    }
    return inv;
}

void task_11() {
    cout << "\n=== 1.1 LU-decomposition ===\n";
    cout << "Size n: ";
    int n; cin >> n;
    Matrix A = readMatrix(n, "A");
    Vector b = readVector(n, "b");

    Matrix L, U; Vector perm; int swaps;
    luDecomposition(A, L, U, perm, swaps);

    Vector x = luSolve(A, b);
    double det = determinantFromLU(A);
    Matrix inv = inverseFromLU(A);

    printMatrix(L, "L");
    printMatrix(U, "U");
    cout << "Row permutation P: [";
    for (int i = 0; i < n; ++i) cout << " " << perm[i];
    cout << " ]\n";
    printVector(x, "x");
    cout << "det(A) = " << setprecision(12) << det << "\n";
    printMatrix(inv, "A^(-1)");
}

Vector thomasSolve(const Vector& a, const Vector& b, const Vector& c, const Vector& d) {
    int n = (int)b.size();
    Vector cc(n, 0.0), dd(n, 0.0), x(n, 0.0);

    if (fabs(b[0]) < EPS_PIVOT) throw runtime_error("Zero first diagonal element");
    cc[0] = c[0] / b[0];
    dd[0] = d[0] / b[0];

    for (int i = 1; i < n; ++i) {
        double denom = b[i] - a[i] * cc[i - 1];
        if (fabs(denom) < EPS_PIVOT) throw runtime_error("Thomas: zero denominator");
        cc[i] = (i < n - 1) ? c[i] / denom : 0.0;
        dd[i] = (d[i] - a[i] * dd[i - 1]) / denom;
    }

    x[n - 1] = dd[n - 1];
    for (int i = n - 2; i >= 0; --i) x[i] = dd[i] - cc[i] * x[i + 1];
    return x;
}

void task_12() {
    cout << "\n=== 1.2 Thomas algorithm ===\n";
    cout << "Size n: ";
    int n; cin >> n;
    cout << "lower (first element unused, " << n << " numbers):\n";
    Vector a = readVector(n, "lower");
    cout << "diag (main diagonal):\n";
    Vector b = readVector(n, "diag");
    cout << "upper (last element unused, " << n << " numbers):\n";
    Vector c = readVector(n, "upper");
    cout << "b (right-hand side):\n";
    Vector d = readVector(n, "b");

    Vector x = thomasSolve(a, b, c, d);
    printVector(x, "x");
}

void toIterForm(const Matrix& A, const Vector& b, Matrix& C, Vector& d) {
    int n = (int)A.size();
    C.assign(n, Vector(n, 0.0));
    d.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        if (fabs(A[i][i]) < EPS_PIVOT) throw runtime_error("Zero on diagonal");
        d[i] = b[i] / A[i][i];
        for (int j = 0; j < n; ++j)
            if (i != j) C[i][j] = -A[i][j] / A[i][i];
    }
}

Vector simpleIteration(const Matrix& A, const Vector& b, double eps, int& iters) {
    Matrix C; Vector d;
    toIterForm(A, b, C, d);
    double q = matNormInf(C);
    int n = (int)A.size();
    Vector x(n, 0.0);

    for (int k = 1; k <= 100000; ++k) {
        Vector xNew(n, 0.0);
        for (int i = 0; i < n; ++i) {
            double s = d[i];
            for (int j = 0; j < n; ++j) s += C[i][j] * x[j];
            xNew[i] = s;
        }
        double delta = diffNormInf(xNew, x);
        iters = k;

        if (q < 1.0) {
            if (q / (1.0 - q) * delta <= eps) return xNew;
        } else if (delta <= eps) return xNew;

        x = xNew;
    }
    throw runtime_error("Simple iteration did not converge");
}

Vector seidel(const Matrix& A, const Vector& b, double eps, int& iters) {
    Matrix C; Vector d;
    toIterForm(A, b, C, d);
    double q = matNormInf(C);
    int n = (int)A.size();
    Vector x(n, 0.0);

    for (int k = 1; k <= 100000; ++k) {
        Vector old = x;
        for (int i = 0; i < n; ++i) {
            double s1 = 0.0, s2 = 0.0;
            for (int j = 0; j < i; ++j) s1 += A[i][j] * x[j];
            for (int j = i + 1; j < n; ++j) s2 += A[i][j] * old[j];
            x[i] = (b[i] - s1 - s2) / A[i][i];
        }
        double delta = diffNormInf(x, old);
        iters = k;

        if (q < 1.0) {
            if (q / (1.0 - q) * delta <= eps) return x;
        } else if (delta <= eps) return x;
    }
    throw runtime_error("Seidel did not converge");
}

void task_13() {
    cout << "\n=== 1.3 Simple iteration & Seidel ===\n";
    cout << "Size n: ";
    int n; cin >> n;
    Matrix A = readMatrix(n, "A");
    Vector b = readVector(n, "b");
    cout << "eps: ";
    double eps; cin >> eps;

    int it1 = 0, it2 = 0;
    Vector x1 = simpleIteration(A, b, eps, it1);
    Vector x2 = seidel(A, b, eps, it2);

    cout << "\nSimple iteration:\n";
    printVector(x1, "x");
    cout << "Iterations: " << it1 << "\n";

    cout << "\nSeidel:\n";
    printVector(x2, "x");
    cout << "Iterations: " << it2 << "\n";
}

void jacobiEigen(Matrix A, double eps, Vector& eigenValues, Matrix& V, int& iters, Vector& history) {
    int n = (int)A.size();
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (fabs(A[i][j] - A[j][i]) > 1e-12)
                throw runtime_error("Jacobi requires symmetric matrix");

    V = identity(n);
    history.clear();

    auto offDiagNorm = [&]() {
        double s = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                s += A[i][j] * A[i][j];
        return sqrt(s);
    };

    history.push_back(offDiagNorm());

    for (int it = 1; it <= 100000; ++it) {
        int p = 0, q = 1;
        double maxVal = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (fabs(A[i][j]) > maxVal) { maxVal = fabs(A[i][j]); p = i; q = j; }

        if (maxVal <= eps) {
            eigenValues.resize(n);
            for (int i = 0; i < n; ++i) eigenValues[i] = A[i][i];
            iters = it - 1;
            return;
        }

        double c, s;
        if (fabs(A[p][p] - A[q][q]) < EPS_PIVOT) {
            c = sqrt(0.5);
            s = (A[p][q] >= 0 ? 1.0 : -1.0) * sqrt(0.5);
        } else {
            double tau = (A[p][p] - A[q][q]) / (2.0 * A[p][q]);
            double t = (tau >= 0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau * tau));
            c = 1.0 / sqrt(1.0 + t * t);
            s = t * c;
        }

        double app = A[p][p], aqq = A[q][q], apq = A[p][q];
        A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A[p][q] = A[q][p] = 0.0;

        for (int k = 0; k < n; ++k) {
            if (k == p || k == q) continue;
            double akp = A[k][p], akq = A[k][q];
            A[k][p] = A[p][k] = c * akp - s * akq;
            A[k][q] = A[q][k] = s * akp + c * akq;
        }

        for (int k = 0; k < n; ++k) {
            double vkp = V[k][p], vkq = V[k][q];
            V[k][p] = c * vkp - s * vkq;
            V[k][q] = s * vkp + c * vkq;
        }

        history.push_back(offDiagNorm());
    }
    throw runtime_error("Jacobi did not converge");
}

void task_14() {
    cout << "\n=== 1.4 Jacobi rotation method ===\n";
    cout << "Size n: ";
    int n; cin >> n;
    Matrix A = readMatrix(n, "A");
    cout << "eps: ";
    double eps; cin >> eps;

    Vector eigenValues, history;
    Matrix V;
    int iters = 0;
    jacobiEigen(A, eps, eigenValues, V, iters, history);

    printVector(eigenValues, "Eigenvalues");
    printMatrix(V, "Eigenvectors (columns)");
    cout << "Rotations: " << iters << "\n";
    cout << "\nOff-diagonal norm by iteration:\n";
    cout << "iter   offdiag_norm\n";
    for (int k = 0; k < (int)history.size(); ++k)
        cout << setw(4) << k << "   " << scientific << setprecision(6) << history[k] << "\n";
}

void qrDecomposition(const Matrix& A, Matrix& Q, Matrix& R) {
    int n = (int)A.size();
    R = A;
    Q = identity(n);

    for (int k = 0; k < n - 1; ++k) {
        Vector x(n - k);
        double normX = 0.0;
        for (int i = k; i < n; ++i) { x[i - k] = R[i][k]; normX += x[i - k] * x[i - k]; }
        normX = sqrt(normX);
        if (normX < EPS_PIVOT) continue;

        double sign = (x[0] >= 0) ? 1.0 : -1.0;
        x[0] += sign * normX;
        double normV = 0.0;
        for (double t : x) normV += t * t;
        normV = sqrt(normV);
        if (normV < EPS_PIVOT) continue;
        for (double& t : x) t /= normV;

        for (int j = k; j < n; ++j) {
            double proj = 0.0;
            for (int i = k; i < n; ++i) proj += x[i - k] * R[i][j];
            for (int i = k; i < n; ++i) R[i][j] -= 2.0 * x[i - k] * proj;
        }

        for (int i = 0; i < n; ++i) {
            double proj = 0.0;
            for (int j = k; j < n; ++j) proj += Q[i][j] * x[j - k];
            for (int j = k; j < n; ++j) Q[i][j] -= 2.0 * proj * x[j - k];
        }
    }
}

Matrix matmul(const Matrix& A, const Matrix& B) {
    int n = (int)A.size();
    int m = (int)B[0].size();
    int p = (int)B.size();
    Matrix C(n, Vector(m, 0.0));
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < p; ++k) {
            double aik = A[i][k];
            if (aik == 0.0) continue;
            for (int j = 0; j < m; ++j) C[i][j] += aik * B[k][j];
        }
    return C;
}

void qrEigenvalues(Matrix A, double eps, vector<complex<double>>& eig, int& iterations) {
    int n = (int)A.size();
    iterations = 0;
    for (int iter = 1; iter <= 10000; ++iter) {
        Matrix Q, R;
        qrDecomposition(A, Q, R);
        A = matmul(R, Q);
        double sub = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < i; ++j) sub = max(sub, fabs(A[i][j]));
        iterations = iter;
        if (sub <= eps) break;
    }

    eig.clear();
    int i = 0;
    while (i < n) {
        if (i == n - 1 || fabs(A[i + 1][i]) <= eps) {
            eig.push_back(complex<double>(A[i][i], 0.0));
            i += 1;
        } else {
            double a = A[i][i], b = A[i][i + 1];
            double c = A[i + 1][i], d = A[i + 1][i + 1];
            double tr = a + d, det = a * d - b * c;
            complex<double> disc(tr * tr - 4.0 * det, 0.0);
            complex<double> root = sqrt(disc);
            eig.push_back((tr + root) / 2.0);
            eig.push_back((tr - root) / 2.0);
            i += 2;
        }
    }
}

void task_15() {
    cout << "\n=== 1.5 QR-decomposition & QR-algorithm ===\n";
    cout << "Size n: ";
    int n; cin >> n;
    Matrix A = readMatrix(n, "A");
    cout << "eps: ";
    double eps; cin >> eps;

    Matrix Q, R;
    qrDecomposition(A, Q, R);
    printMatrix(Q, "Q");
    printMatrix(R, "R");

    Matrix QR = matmul(Q, R);
    printMatrix(QR, "Check Q*R");

    vector<complex<double>> eig;
    int iters = 0;
    qrEigenvalues(A, eps, eig, iters);

    cout << "\nEigenvalues:\n";
    for (size_t k = 0; k < eig.size(); ++k) {
        cout << "  lambda_" << k + 1 << " = "
             << eig[k].real();
        if (eig[k].imag() >= 0) cout << "+";
        cout << eig[k].imag() << "i\n";
    }
    cout << "QR iterations: " << iters << "\n";
}

int main() {
    cout << fixed << setprecision(6);

    while (true) {
        cout << "\n" << string(60, '=') << "\n";
        cout << "LAB 1 - NUMERICAL METHODS OF LINEAR ALGEBRA\n";
        cout << string(60, '=') << "\n";
        cout << "1 - 1.1 LU-decomposition (solve SLAE, det, inverse)\n";
        cout << "2 - 1.2 Thomas algorithm\n";
        cout << "3 - 1.3 Simple iteration & Seidel\n";
        cout << "4 - 1.4 Jacobi rotation method\n";
        cout << "5 - 1.5 QR-decomposition & QR-algorithm\n";
        cout << "0 - Exit\n";
        cout << "Choice: ";

        int choice;
        if (!(cin >> choice)) {
            cin.clear();
            clearInput();
            continue;
        }

        try {
            if (choice == 1) task_11();
            else if (choice == 2) task_12();
            else if (choice == 3) task_13();
            else if (choice == 4) task_14();
            else if (choice == 5) task_15();
            else if (choice == 0) { cout << "Done.\n"; break; }
            else cout << "Unknown option.\n";
        } catch (const exception& e) {
            cout << "\nError: " << e.what() << "\n";
        }

        cout << "\nPress Enter to return to menu...";
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cin.get();
    }
    return 0;
}
