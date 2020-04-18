#include "ukf.h"
#include "Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  n_x_ = 5;

  n_aug_ = n_x_ + 2;
  lambda_ = 3 - n_aug_;
  int num_sigma_points = 2 * n_aug_ + 1;

  x_ << 0, 0, 0, 0, 0;
  P_ << 0.0043, -0.0013, 0.0030, -0.0022, -0.0020, -0.0013, 0.0077, 0.0011, 0.0071, 0.0060, 0.0030, 0.0011, 0.0054,
      0.0007, 0.0008, -0.0022, 0.0071, 0.0007, 0.0098, 0.0100, -0.0020, 0.0060, 0.0008, 0.0100, 0.0123;
  Xsig_pred_ = MatrixXd(n_aug_, num_sigma_points);
}

UKF::~UKF()
{
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Make sure you switch between lidar and
   * radar
   * measurements.
   */
  if (meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    UpdateLidar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(meas_package);
  }
}

void UKF::Prediction(double delta_t)
{
  /**
   * TODO: Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */
  Eigen::MatrixXd Xsig_aug = CreateAugmentedMatrix();
  CreateSigmaPoints(Xsig_aug, delta_t);

  // Predict Mean and Covariance
  // Predict Mean
  PredictMean();
  PredictCovariance();
}
Eigen::MatrixXd UKF::CreateAugmentedMatrix()
{
  // create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);
  // create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  MatrixXd A = P_aug.llt().matrixL();
  MatrixXd sig_matrix = A * std::sqrt((lambda_ + n_x_));
  Eigen::MatrixXd Xsig_aug(n_aug_, 2 * n_aug_ + 1);

  // create sigma point matrix
  MatrixXd sigma_matrix = A * std::sqrt(lambda_ + n_aug_);
  Xsig_aug.col(0) = x_aug;

  for (int col_no = 0; col_no < n_aug_; ++col_no)
  {
    Xsig_aug.col(col_no + 1) = x_aug + sigma_matrix.col(col_no);
    Xsig_aug.col(col_no + n_aug_ + 1) = x_aug - sigma_matrix.col(col_no);
  }
  return Xsig_aug;
}

void UKF::PredictMean()
{
  // Predict Mean
  VectorXd weights = VectorXd(2 * n_aug_ + 1);
  double w_major = lambda_ / (lambda_ + n_aug_);
  double w_minor = 0.5 / (lambda_ + n_aug_);

  weights.fill(w_minor);
  weights(0) = w_major;

  for (int col_no = 0; col_no < Xsig_pred_.cols(); ++col_no)
  {
    x_ = x_ + (weights(col_no) * Xsig_pred_.col(col_no));
  }
}

VectorXd UKF::GetWeights()
{
  VectorXd weights = VectorXd(2 * n_aug_ + 1);
  double w_major = lambda_ / (lambda_ + n_aug_);
  double w_minor = 0.5 / (lambda_ + n_aug_);

  weights.fill(w_minor);
  weights(0) = w_major;
  return weights;
}

void UKF::PredictCovariance()
{
  // Predict Mean
  VectorXd weights = VectorXd(2 * n_aug_ + 1);
  double w_major = lambda_ / (lambda_ + n_aug_);
  double w_minor = 0.5 / (lambda_ + n_aug_);

  weights.fill(w_minor);
  weights(0) = w_major;

  // predict state covariance matrix
  P_.fill(0.0);  // TODO Is this right?
  for (int col_no = 0; col_no < Xsig_pred_.cols(); ++col_no)
  {
    VectorXd residual = Xsig_pred_.col(col_no) - x_;
    // angle normalization
    while (residual(3) > M_PI)
    {
      residual(3) -= 2. * M_PI;
    }
    while (residual(3) < -M_PI)
    {
      residual(3) += 2. * M_PI;
    }
    P_ = P_ + (weights(col_no) * (residual * residual.transpose()));
  }
}

void UKF::CreateSigmaPoints(Eigen::MatrixXd Xsig_aug, double delta_t)
{
  for (int col_no = 0; col_no < Xsig_aug.cols(); ++col_no)
  {
    double p_x = Xsig_aug(0, col_no);
    double p_y = Xsig_aug(1, col_no);
    double v = Xsig_aug(2, col_no);
    double psi = Xsig_aug(3, col_no);
    double psi_dot = Xsig_aug(4, col_no);
    double nu = Xsig_aug(5, col_no);
    double nu_dot_dot = Xsig_aug(6, col_no);

    // avoid division by zero

    if (fabs(psi) < 0.001)
    {
      Xsig_pred_(0, col_no) = p_x + v * std::cos(psi) * delta_t + 0.5 * delta_t * delta_t * std::cos(psi) * nu;
      Xsig_pred_(1, col_no) = p_y + v * std::sin(psi) * delta_t + 0.5 * delta_t * delta_t * std::sin(psi) * nu;
    }
    else
    {
      double factor = v / psi_dot;
      Xsig_pred_(0, col_no) = p_x + factor * (std::sin(psi + psi_dot * delta_t) - std::sin(psi)) +
                              0.5 * delta_t * delta_t * std::cos(psi) * nu;
      Xsig_pred_(1, col_no) = p_y + factor * (-std::cos(psi + psi_dot * delta_t) + std::cos(psi)) +
                              0.5 * delta_t * delta_t * std::sin(psi) * nu;
    }
    Xsig_pred_(2, col_no) = v + 0 + delta_t * nu;
    Xsig_pred_(3, col_no) = psi + psi_dot * delta_t + 0.5 * delta_t * delta_t * nu_dot_dot;
    Xsig_pred_(4, col_no) = psi_dot + delta_t * nu_dot_dot;
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  int n_z = 3;
  VectorXd z = meas_package.raw_measurements_;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  Eigen::VectorXd weights = GetWeights();

  // transform sigma points into measurement space
  for (int col_no = 0; col_no < Xsig_pred_.cols(); ++col_no)
  {
    double px = Xsig_pred_(0, col_no);
    double py = Xsig_pred_(1, col_no);
    double v = Xsig_pred_(2, col_no);
    double psi = Xsig_pred_(3, col_no);

    Zsig(0, col_no) = std::sqrt(px * px + py * py);
    Zsig(1, col_no) = std::atan(py / px);
    Zsig(2, col_no) = (px * std::cos(psi) * v + py * std::sin(psi) * v) / (std::sqrt(px * px + py * py));
  }

  // calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int col_no = 0; col_no < Xsig_pred_.cols(); ++col_no)
  {
    z_pred = z_pred + weights(col_no) * Zsig.col(col_no);
  }

  // calculate innovation covariance matrix S
  S.fill(0.0);
  MatrixXd R(n_z, n_z);
  R.fill(0.0);
  R(0, 0) = std_radr_ * std_radr_;
  R(1, 1) = std_radphi_ * std_radphi_;
  R(2, 2) = std_radrd_ * std_radrd_;
  for (int col_no = 0; col_no < Xsig_pred_.cols(); ++col_no)
  {
    VectorXd residual = Zsig.col(col_no) - z_pred;
    S = S + weights(col_no) * (residual * residual.transpose());
  }
  S = S + R;

  // Update state
  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // /**
  //  * Student part begin
  //  */

  Tc.fill(0.0);
  // calculate cross correlation matrix
  std::cout << Xsig_pred_ << std::endl;
  // Eigen::MatrixXd X_sig_pred_ss = Xsig_pred_.block(0, 0, 4, 14);  // TODO remove hardcode
  for (int col_no = 0; col_no < Zsig.cols(); ++col_no)
  {
    VectorXd state_residual = X_sig_pred_ss.col(col_no) - x_;
    VectorXd measurement_residual = Zsig.col(col_no) - z_pred;
    Tc = Tc + weights(col_no) * (state_residual * measurement_residual.transpose());
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  // update state mean and covariance matrix

  x_ = x_ + K * (z - z_pred);
  P_ = P_ - K * S * K.transpose();
}
