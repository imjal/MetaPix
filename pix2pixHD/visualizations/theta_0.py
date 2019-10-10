from golden_testset import MetaPixTest, FileType
str_base = '/home/jl5/data/data-meta/experiments/'
dirs_theta_0 = {
    "PT":str_base+"144_TEST_THETA0_PT", 
    "Theta_0 10/10":str_base+"139_TEST_THETA0_1010", 
    "Theta_0 20/20":str_base+"142_TEST_THETA0_2020", 
    "theta_0 50/50": str_base+"141_TEST_THETA0_5050", 
    "theta_0 100100": str_base +"140_TEST_THETA0_100100", 
    "100100 FT": str_base + "072_MT_PT_EP1_100100/finetune_300"
}
mt0 = MetaPixTest(dirs_theta_0, FileType.png, "/home/jl5/data/data-meta/", "/home/jl5/data/data-meta/experiments/143_VIZ_THETA0_1/", debug=False)
mt0.create_test_set()
mt0.generate_visualization()