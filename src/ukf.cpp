#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
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
  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_x_;

  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(1 / (2 * (lambda_ + n_aug_)));
  weights_(0) = lambda_ / (lambda_ + n_aug_);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  long long  time_us_update;

  if (is_initialized_ == false) 
  {
    // process unitialization according the sensor type.
    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR) 
    {
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rho_dot = meas_package.raw_measurements_(2);
      double x = rho * cos(phi);
      double y = rho * sin(phi);
      double v = rho_dot;
      double vx = rho_dot * cos(phi);
      double vy = rho_dot * sin(phi);

      x_ << x, y, v, rho, rho_dot;
      P_ << std_radr_ * std_radr_,    0, 0, 0, 0,
            0, std_radr_ * std_radr_, 0, 0, 0,
            0, 0, std_radrd_ * std_radrd_, 0, 0,
            0, 0, 0, std_radphi_ * std_radphi_, 0,
            0, 0, 0, 0, std_radphi_ * std_radphi_;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER)
    {
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
      P_ << std_laspx_ * std_laspx_, 0, 0, 0, 0,
            0, std_laspy_ * std_laspy_, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;
    } 

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
  }
  else
  {
    // compute the time elapsed between the current and previous measurements in seconds.
    time_us_update = meas_package.timestamp_;
    float delta_t = (time_us_update - time_us_) / 1000000.0;
    time_us_ =time_us_update;

    // Predict the next states and covariance matrix
    Prediction(delta_t);

    // Update the next states and covariance matrix
    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR) 
    {
      UpdateRadar(meas_package);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) 
    {
      UpdateLidar(meas_package);
    } 
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
 
  // create augmented mean state
  x_aug.setZero(n_aug_);
  x_aug.head(5) = x_;

  // create augmented covariance matrix
  P_aug.setZero(n_aug_, n_aug_);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug.bottomRightCorner(2, 2) << std_a_ * std_a_,           0, 
                                         0,       std_yawdd_ * std_yawdd_;
  
  // create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  // create augmented sigma points
  MatrixXd A1 = MatrixXd(n_aug_, n_aug_);
  MatrixXd A2 = MatrixXd(n_aug_, n_aug_);
  A1 = std::sqrt(lambda_ + n_aug_) * A;
  A2 = std::sqrt(lambda_ + n_aug_) * A;
  
  Xsig_aug.col(0)   = x_aug;
  Xsig_aug.col(1)   = x_aug + A1.col(0);
  Xsig_aug.col(2)   = x_aug + A1.col(1);
  Xsig_aug.col(3)   = x_aug + A1.col(2);
  Xsig_aug.col(4)   = x_aug + A1.col(3);
  Xsig_aug.col(5)   = x_aug + A1.col(4);
  Xsig_aug.col(6)   = x_aug + A1.col(5);
  Xsig_aug.col(7)   = x_aug + A1.col(6);
  
  Xsig_aug.col(8)   = x_aug - A2.col(0);
  Xsig_aug.col(9)   = x_aug - A2.col(1);
  Xsig_aug.col(10)  = x_aug - A2.col(2);
  Xsig_aug.col(11)  = x_aug - A2.col(3);
  Xsig_aug.col(12)  = x_aug - A2.col(4);
  Xsig_aug.col(13)  = x_aug - A2.col(5);
  Xsig_aug.col(14)  = x_aug - A2.col(6);

  // Predict sigma points
  // create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  float px = 0;
  float py = 0;
  float v = 0;
  float yaw = 0;
  float yaw_rate = 0;
  float nu_accel = 0;
  float nu_yaw_accel = 0;
  
  VectorXd vec(n_x_);
  VectorXd noise_vec(n_x_);
  
  for (int i = 0; i < Xsig_aug.cols(); i++) 
  {
    px           = Xsig_aug(0, i);
    py           = Xsig_aug(1, i);
    v            = Xsig_aug(2, i);
    yaw          = Xsig_aug(3, i);
    yaw_rate     = Xsig_aug(4, i);
    nu_accel     = Xsig_aug(5, i);
    nu_yaw_accel = Xsig_aug(6, i);
                 
    // avoid division by zero                          
    if ( std::abs( yaw_rate ) <= 0.001 ) 
    {
      vec << v * std::cos(yaw) * delta_t, v * std::sin(yaw) * delta_t, 0, 0, 0;
    } 
    else 
    {
      vec << (v / yaw_rate) * (std::sin(yaw + yaw_rate * delta_t) - std::sin(yaw)),
             (v / yaw_rate) * (-std::cos(yaw + yaw_rate * delta_t) + std::cos(yaw)),
             0, 
             yaw_rate * delta_t, 
             0;
    }

    //noise
    noise_vec << (1/2) * delta_t * delta_t * std::cos(yaw) * nu_accel,
                 (1/2) * delta_t * delta_t * std::sin(yaw) * nu_accel,
                 delta_t * nu_accel, 
                 (1/2) * delta_t * delta_t * nu_yaw_accel,
                 delta_t * nu_yaw_accel;

    // write predicted sigma points into right column
    Xsig_pred_.col(i) = Xsig_aug.col(i).head(n_x_) + vec + noise_vec;
  }

  // predict state mean
  //x.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  {  // iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // predict state covariance matrix
  //P.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  {  // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  // The LiDAR noise profile is linear, so we can use the standard linear 
  // Kalman filter here.

  // measurement noise covariance matrix
  MatrixXd R = MatrixXd(2,2);

  // New - define measurement matrix
  MatrixXd H;
  H.setZero(2, n_x_);
  H(0, 0) = H(1, 1) = 1;  // Select out the position elements only
  VectorXd z_pred = H * x_;
  const VectorXd z = meas_package.raw_measurements_;

  // Calculate residual vector z_diff
  const VectorXd z_diff = z - z_pred;

  // New - define measurement noise matrix
  R.setZero(2, 2);
  R << std_laspx_ * std_laspx_, 0, 0, std_laspy_ * std_laspy_;

  // Create innovation covariance matrix S
  MatrixXd S = H * P_ * H.transpose() + R;

  // Create Kalman gain matrix K
  MatrixXd K = P_ * H.transpose() * S.inverse();

  // Create new estimate for states and covariance
  x_ = x_ + (K * z_diff);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H) * P_;

  // Calculate NIS for LiDAR
  NIS_lidar_ = z_diff.transpose() * S.inverse() * z_diff;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
   // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  // measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) // 2n+1 simga points
  {  
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                       // r
    Zsig(1,i) = atan2(p_y,p_x);                                // phi
    Zsig(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot
  }

  // mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; ++i) 
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) // 2n+1 simga points
  {  
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  R <<  std_radr*std_radr, 0, 0,
        0, std_radphi*std_radphi, 0,
        0, 0,std_radrd*std_radrd;
  S = S + R;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)   // 2n+1 simga points
  {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  
  // residual
  const VectorXd z = meas_package.raw_measurements_;
  VectorXd z_diff = z - z_pred;

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // Calculate NIS for Radar
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}