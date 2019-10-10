from golden_testset import MetaPixTest, FileType
str_base = '/home/jl5/data/data-meta/experiments/'
dirs_new_fig = {
    "Theta 0 - MT  100/100": str_base + "140_TEST_THETA0_100100",
    "10 MT": str_base + "176_TEST_ALL_MT_VIZ/10", 
    "20 MT": str_base + "176_TEST_ALL_MT_VIZ/20", 
    "40 MT": str_base + "176_TEST_ALL_MT_VIZ/40", 
    "80 MT": str_base + "176_TEST_ALL_MT_VIZ/80", 
    "200 MT": str_base + "176_TEST_ALL_MT_VIZ/200",
    "Theta 0 - PT  100/100": str_base + "144_TEST_THETA0_PT",
    "10 PT": str_base + "178_TEST_ALL_MT_VIZ/10", 
    "20 PT": str_base + "178_TEST_ALL_MT_VIZ/20", 
    "40 PT": str_base + "178_TEST_ALL_MT_VIZ/40", 
    "80 PT": str_base + "178_TEST_ALL_MT_VIZ/80",
    "200 PT": str_base + "178_TEST_ALL_MT_VIZ/200"
}
m_new_fig = MetaPixTest(dirs_new_fig, FileType.png, "/home/jl5/data/data-meta/", "/home/jl5/data/data-meta/experiments/186_VIZ_newfig/", debug=False)
m_new_fig.create_test_set()
m_new_fig.generate_visualization(subset=True, num_output_imgs=5)