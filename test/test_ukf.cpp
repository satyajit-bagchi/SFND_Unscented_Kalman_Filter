#define private public

#include <gtest/gtest.h>
#include "../src/ukf.cpp"
#include "../src/Eigen/Dense"

class UKFTest : public ::testing::Test
{
public:
  UKF ukf;

  void SetUp() override
  {
    ukf.std_a_ = 0.2;
    ukf.std_yawdd_ = 0.2;
  }
};

TEST(TestUKF, TestInit)
{
  UKF ukf;
}

TEST_F(UKFTest, GetWeights)
{
  Eigen::VectorXd weights = ukf.GetWeights();
  ASSERT_EQ(weights.size(), 15);
  ASSERT_EQ(weights(1), weights(2));
}

TEST_F(UKFTest, TestCreateAugmentedMatrix)
{
  ukf.std_a_ = 0.2;
  ukf.std_yawdd_ = 0.2;
  ukf.x_ << 5.7441, 1.3800, 2.2049, 0.5015, 0.3528;
  ukf.P_ << 0.0043, -0.0013, 0.0030, -0.0022, -0.0020, -0.0013, 0.0077, 0.0011, 0.0071, 0.0060, 0.0030, 0.0011, 0.0054,
      0.0007, 0.0008, -0.0022, 0.0071, 0.0007, 0.0098, 0.0100, -0.0020, 0.0060, 0.0008, 0.0100, 0.0123;

  Eigen::MatrixXd P_aug = MatrixXd(7, 7);

  // create sigma point matrix
  Eigen::MatrixXd Xsig_aug = MatrixXd(7, 2 * 7 + 1);
  Xsig_aug << 5.7441, 5.85768, 5.7441, 5.7441, 5.7441, 5.7441, 5.7441, 5.7441, 5.63052, 5.7441, 5.7441, 5.7441, 5.7441,
      5.7441, 5.7441, 1.38, 1.34566, 1.52806, 1.38, 1.38, 1.38, 1.38, 1.38, 1.41434, 1.23194, 1.38, 1.38, 1.38, 1.38,
      1.38, 2.2049, 2.28414, 2.24557, 2.29582, 2.2049, 2.2049, 2.2049, 2.2049, 2.12566, 2.16423, 2.11398, 2.2049,
      2.2049, 2.2049, 2.2049, 0.5015, 0.44339, 0.631886, 0.516923, 0.595227, 0.5015, 0.5015, 0.5015, 0.55961, 0.371114,
      0.486077, 0.407773, 0.5015, 0.5015, 0.5015, 0.3528, 0.299973, 0.462123, 0.376339, 0.48417, 0.418721, 0.3528,
      0.3528, 0.405627, 0.243477, 0.329261, 0.22143, 0.286879, 0.3528, 0.3528, 0, 0, 0, 0, 0, 0, 0.34641, 0, 0, 0, 0, 0,
      0, -0.34641, 0, 0, 0, 0, 0, 0, 0, 0, 0.34641, 0, 0, 0, 0, 0, 0, -0.34641;
  Eigen::MatrixXd output = ukf.CreateAugmentedMatrix();

  std::cout << output;
  EXPECT_TRUE(Xsig_aug.isApprox(output, 1e-2));
}

TEST_F(UKFTest, CreateSigmaPoints)
{
  //   ukf.P_ << 0.0043, -0.0013, 0.0030, -0.0022, -0.0020, -0.0013, 0.0077, 0.0011, 0.0071, 0.0060, 0.0030, 0.0011,
  //   0.0054,
  //       0.0007, 0.0008, -0.0022, 0.0071, 0.0007, 0.0098, 0.0100, -0.0020, 0.0060, 0.0008, 0.0100, 0.0123;
  //   ukf.x_ << 5.7441, 1.3800, 2.2049, 0.5015, 0.3528;
  //   Eigen::MatrixXd expected_Xsig_pred(5, 15);

  //   expected_Xsig_pred << 5.7441, 5.85768, 5.7441, 5.7441, 5.7441, 5.7441, 5.63052, 5.7441, 5.7441, 5.7441, 5.7441,
  //   1.38,
  //       1.34566, 1.52806, 1.38, 1.38, 1.38, 1.41434, 1.23194, 1.38, 1.38, 1.38, 2.2049, 2.28414, 2.24557, 2.29582,
  //       2.2049,
  //       2.2049, 2.12566, 2.16423, 2.11398, 2.2049, 2.2049, 0.5015, 0.44339, 0.631886, 0.516923, 0.595227, 0.5015,
  //       0.55961,
  //       0.371114, 0.486077, 0.407773, 0.5015, 0.3528, 0.299973, 0.462123, 0.376339, 0.48417, 0.418721, 0.405627,
  //       0.243477,
  //       0.329261, 0.22143, 0.286879;
}

TEST_F(UKFTest, TestPredictMean)
{
  // set state dimension
  int n_x = 5;

  // set augmented dimension
  int n_aug = 7;
  //   MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);
  //   ukf.Xsig_pred_ << 5.9374, 6.0640, 5.925, 5.9436, 5.9266, 5.9374, 5.9389, 5.9374, 5.8106, 5.9457, 5.9310, 5.9465,
  //       5.9374, 5.9359, 5.93744, 1.48, 1.4436, 1.660, 1.4934, 1.5036, 1.48, 1.4868, 1.48, 1.5271, 1.3104, 1.4787,
  //       1.4674,
  //       1.48, 1.4851, 1.486, 2.204, 2.2841, 2.2455, 2.2958, 2.204, 2.204, 2.2395, 2.204, 2.1256, 2.1642, 2.1139,
  //       2.204,
  //       2.204, 2.1702, 2.2049, 0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337, 0.5367, 0.53851, 0.60017,
  //       0.39546,
  //       0.51900, 0.42991, 0.530188, 0.5367, 0.535048, 0.352, 0.29997, 0.46212, 0.37633, 0.4841, 0.41872, 0.352,
  //       0.38744,
  //       0.40562, 0.24347, 0.32926, 0.2214, 0.28687, 0.352, 0.318159;

  //   ukf.PredictMean();
  //   Eigen::MatrixXd mean;
  //   mean << 5.93637, 1.49035, 2.20528, 0.536853, 0.353577;
  //   EXPECT_EQ(ukf.x_, mean);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}