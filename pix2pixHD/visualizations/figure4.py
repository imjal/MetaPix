from golden_testset import MetaPixTest, FileType
str_base = '/home/jl5/data/data-meta/experiments/'
dirs = {
    "finetune inf inf PT": str_base+"180_TEST_071_ALL_IMAGES",
    "finetune inf inf RD": str_base + "179_TEST_110_ALL_IMAGES",
    "PW FT inf inf PT": str_base + "185_FT_PT_AUTH_inf",
    "Pix2Pix PT k=5 T=10/10":str_base + "181_TEST_065_1010PT",
    "MetaPix Pix2Pix k=5 T=10/10":str_base+"138_TEST_62",
    "finetune PW rd 5 20": str_base+"164_FT_RD_k5T20",
    "MT": str_base + "173_PW_MT_k5_T20_ADD",
    "PT": str_base + '174_PW_PT_k5_T20_ADD'
}
m_tbl = MetaPixTest(dirs, FileType.png, "/home/jl5/data/data-meta/", "/home/jl5/data/data-meta/experiments/177_Table_Figure/", debug=False)
m_tbl.create_test_set()
m_tbl.generate_visualization()
#mean_scores = m_tbl.test_images()
#print(mean_scores)
