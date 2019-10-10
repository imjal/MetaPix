from golden_testset import MetaPixTest, FileType
str_base = '/home/jl5/data/data-meta/experiments/'
dirs = {
    "Theta 0 - MT  100/100": str_base + "140_TEST_THETA0_100100",
    "10 MT": str_base + "176_TEST_ALL_MT_VIZ/10", 
    "20 MT": str_base + "176_TEST_ALL_MT_VIZ/20", 
    "30 MT": str_base + "176_TEST_ALL_MT_VIZ/30", 
    "40 MT": str_base + "176_TEST_ALL_MT_VIZ/40", 
    "50 MT": str_base + "176_TEST_ALL_MT_VIZ/50", 
    "80 MT": str_base + "176_TEST_ALL_MT_VIZ/80", 
    "100 MT": str_base + "176_TEST_ALL_MT_VIZ/100", 
    "200 MT": str_base + "176_TEST_ALL_MT_VIZ/200",
    "Theta 0 - PT  100/100": str_base + "144_TEST_THETA0_PT",
    "10 PT": str_base + "178_TEST_ALL_MT_VIZ/10", 
    "20 PT": str_base + "178_TEST_ALL_MT_VIZ/20", 
    "30 PT": str_base + "178_TEST_ALL_MT_VIZ/30", 
    "40 PT": str_base + "178_TEST_ALL_MT_VIZ/40", 
    "50 PT": str_base + "178_TEST_ALL_MT_VIZ/50", 
    "80 PT": str_base + "178_TEST_ALL_MT_VIZ/80", 
    "100 PT": str_base + "178_TEST_ALL_MT_VIZ/100", 
    "200 PT": str_base + "178_TEST_ALL_MT_VIZ/200"
}
m = MetaPixTest(dirs, FileType.png, "/home/jl5/data/data-meta/", "/home/jl5/data/data-meta/experiments/177_VIZ_FTLOG/", debug=True)
m.create_test_set()
m.generate_visualization()