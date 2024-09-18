#include <gtest/gtest.h>
#include "BucketGraph.h"

// Mock classes and data to support the tests
class MockLabel : public Label {
public:
    MockLabel(int job_id, double cost, std::vector<double> resources)
        : Label(job_id, cost, resources) {}
};

class MockVRPJob : public VRPJob {
public:
    MockVRPJob(int id, double duration, double demand)
        : VRPJob(id, duration, demand) {}
};

// Test fixture for BucketGraph
class BucketGraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize mock data
        jobs = {
            MockVRPJob(0, 10.0, 5.0),
            MockVRPJob(1, 15.0, 10.0),
            MockVRPJob(2, 20.0, 15.0)
        };

        distance_matrix = {
            {0.0, 10.0, 20.0},
            {10.0, 0.0, 15.0},
            {20.0, 15.0, 0.0}
        };

        bucketGraph = std::make_unique<BucketGraph>(jobs, 100, 10);
        bucketGraph->set_distance_matrix(distance_matrix);
    }

    std::vector<MockVRPJob> jobs;
    std::vector<std::vector<double>> distance_matrix;
    std::unique_ptr<BucketGraph> bucketGraph;
};

// Test for initInfo method
TEST_F(BucketGraphTest, InitInfoTest) {
    // Redirect stdout to capture the output
    testing::internal::CaptureStdout();
    bucketGraph->initInfo();
    std::string output = testing::internal::GetCapturedStdout();

    // Check if the output contains expected configuration info
    EXPECT_NE(output.find("CONFIGURATION INFO"), std::string::npos);
    EXPECT_NE(output.find("Resources"), std::string::npos);
    EXPECT_NE(output.find("Number of Clients"), std::string::npos);
    EXPECT_NE(output.find("Maximum SRC cuts"), std::string::npos);
}

// Test for is_job_visited and set_job_visited methods
TEST_F(BucketGraphTest, JobVisitedTest) {
    std::array<uint64_t, 1> bitmap = {0};

    // Initially, no job should be visited
    EXPECT_FALSE(bucketGraph->is_job_visited(bitmap, 0));
    EXPECT_FALSE(bucketGraph->is_job_visited(bitmap, 1));

    // Mark job 0 as visited
    bucketGraph->set_job_visited(bitmap, 0);
    EXPECT_TRUE(bucketGraph->is_job_visited(bitmap, 0));
    EXPECT_FALSE(bucketGraph->is_job_visited(bitmap, 1));

    // Mark job 1 as visited
    bucketGraph->set_job_visited(bitmap, 1);
    EXPECT_TRUE(bucketGraph->is_job_visited(bitmap, 0));
    EXPECT_TRUE(bucketGraph->is_job_visited(bitmap, 1));
}

// Test for is_job_unreachable and set_job_unreachable methods
TEST_F(BucketGraphTest, JobUnreachableTest) {
    std::array<uint64_t, 1> bitmap = {0};

    // Initially, no job should be unreachable
    EXPECT_FALSE(bucketGraph->is_job_unreachable(bitmap, 0));
    EXPECT_FALSE(bucketGraph->is_job_unreachable(bitmap, 1));

    // Mark job 0 as unreachable
    bucketGraph->set_job_unreachable(bitmap, 0);
    EXPECT_TRUE(bucketGraph->is_job_unreachable(bitmap, 0));
    EXPECT_FALSE(bucketGraph->is_job_unreachable(bitmap, 1));

    // Mark job 1 as unreachable
    bucketGraph->set_job_unreachable(bitmap, 1);
    EXPECT_TRUE(bucketGraph->is_job_unreachable(bitmap, 0));
    EXPECT_TRUE(bucketGraph->is_job_unreachable(bitmap, 1));
}

// Test for check_feasibility method
TEST_F(BucketGraphTest, CheckFeasibilityTest) {
    MockLabel fw_label(0, 10.0, {30.0, 5.0});
    MockLabel bw_label(1, 15.0, {60.0, 10.0});

    // Check feasibility between forward and backward labels
    EXPECT_TRUE(bucketGraph->check_feasibility(&fw_label, &bw_label));

    // Modify resources to make it infeasible
    bw_label.resources[0] = 40.0;
    EXPECT_FALSE(bucketGraph->check_feasibility(&fw_label, &bw_label));
}

// Test for print_statistics method
TEST_F(BucketGraphTest, PrintStatisticsTest) {
    // Redirect stdout to capture the output
    testing::internal::CaptureStdout();
    bucketGraph->print_statistics();
    std::string output = testing::internal::GetCapturedStdout();

    // Check if the output contains expected statistics info
    EXPECT_NE(output.find("Metric"), std::string::npos);
    EXPECT_NE(output.find("Forward"), std::string::npos);
    EXPECT_NE(output.find("Backward"), std::string::npos);
    EXPECT_NE(output.find("Labels"), std::string::npos);
    EXPECT_NE(output.find("Dominance Check"), std::string::npos);
}

// Test for setup method
TEST_F(BucketGraphTest, SetupTest) {
    bucketGraph->setup();

    // Check if fixed_arcs and fixed_buckets are initialized correctly
    EXPECT_EQ(bucketGraph->fixed_arcs.size(), jobs.size());
    EXPECT_EQ(bucketGraph->fw_fixed_buckets.size(), bucketGraph->fw_buckets.size());
    EXPECT_EQ(bucketGraph->bw_fixed_buckets.size(), bucketGraph->bw_buckets.size());

    for (const auto &row : bucketGraph->fixed_arcs) {
        EXPECT_EQ(row.size(), jobs.size());
    }

    for (const auto &fb : bucketGraph->fw_fixed_buckets) {
        EXPECT_EQ(fb.size(), bucketGraph->fw_buckets.size());
    }

    for (const auto &bb : bucketGraph->bw_fixed_buckets) {
        EXPECT_EQ(bb.size(), bucketGraph->bw_buckets.size());
    }
}

// Test for redefine method
TEST_F(BucketGraphTest, RedefineTest) {
    int new_bucket_interval = 20;
    bucketGraph->redefine(new_bucket_interval);

    // Check if bucket interval is updated correctly
    EXPECT_EQ(bucketGraph->bucket_interval, new_bucket_interval);

    // Check if intervals are reinitialized correctly
    EXPECT_EQ(bucketGraph->intervals.size(), R_SIZE);
    for (const auto &interval : bucketGraph->intervals) {
        EXPECT_EQ(interval.size, new_bucket_interval);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}