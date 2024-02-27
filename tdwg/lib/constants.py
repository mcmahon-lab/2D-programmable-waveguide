class Prism():
    def __init__(self, n_e, n_o):
        self.n_e = n_e
        self.n_o = n_o
    
    def __str__(self):
        return f'Prism ordinary index {self.n_o:.3f}, extraordinary index {self.n_e:.2f}'

# from https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=3243
Rutile_1550 = Prism(2.691, 2.449)
Rutile_633 = Prism(2.874, 2.583)