__author__ = 'Lee'

# -*- coding: utf-8 -*-

import geopandas as gpd
import shutil
import os
from osgeo import gdal
from osgeo import ogr
import shapefile

def affine_fit(from_pts, to_pts):
    q = from_pts
    p = to_pts
    if len(q) != len(p) or len(q) < 1:
        print("原始点和目标点的个数必须相同.")
        return False

    dim = len(q[0])  # 维度
    if len(q) < dim:
        print("点数小于维度.")
        return False

    # 新建一个空的 维度 x (维度+1) 矩阵 并填满
    c = [[0.0 for a in range(dim)] for i in range(dim + 1)]
    for j in range(dim):
        for k in range(dim + 1):
            for i in range(len(q)):
                qt = list(q[i]) + [1]
                c[k][j] += qt[k] * p[i][j]

    # 新建一个空的 (维度+1) x (维度+1) 矩阵 并填满
    Q = [[0.0 for a in range(dim)] + [0] for i in range(dim + 1)]
    for qi in q:
        qt = list(qi) + [1]
        for i in range(dim + 1):
            for j in range(dim + 1):
                Q[i][j] += qt[i] * qt[j]

    # 判断原始点和目标点是否共线，共线则无解. 耗时计算，如果追求效率可以不用。
    # 其实就是解n个三元一次方程组
    def gauss_jordan(m, eps=1.0 / (10 ** 10)):
        (h, w) = (len(m), len(m[0]))
        for y in range(0, h):
            maxrow = y
            for y2 in range(y + 1, h):
                if abs(m[y2][y]) > abs(m[maxrow][y]):
                    maxrow = y2
            (m[y], m[maxrow]) = (m[maxrow], m[y])
            if abs(m[y][y]) <= eps:
                return False
            for y2 in range(y + 1, h):
                c = m[y2][y] / m[y][y]
                for x in range(y, w):
                    m[y2][x] -= m[y][x] * c
        for y in range(h - 1, 0 - 1, -1):
            c = m[y][y]
            for y2 in range(0, y):
                for x in range(w - 1, y - 1, -1):
                    m[y2][x] -= m[y][x] * m[y2][y] / c
            m[y][y] /= c
            for x in range(h, w):
                m[y][x] /= c
        return True

    M = [Q[i] + c[i] for i in range(dim + 1)]
    if not gauss_jordan(M):
        print("错误，原始点和目标点也许是共线的.")
        return False

    class transformation:
        """对象化仿射变换."""

        def To_Str(self):
            res = ""
            for j in range(dim):
                str = "x%d' = " % j
                for i in range(dim):
                    str += "x%d * %f + " % (i, M[i][j + dim + 1])
                str += "%f" % M[dim][j + dim + 1]
                res += str + "\n"
            return res

        def transform(self, pt):
            res = [0.0 for a in range(dim)]
            for j in range(dim):
                for i in range(dim):
                    res[j] += pt[i] * M[i][j + dim + 1]
                res[j] += M[dim][j + dim + 1]
            return res

    return transformation()


def transform_shp(same_point, shp_path, res_path):
    road_tran = affine_fit(same_point[['x_before', 'y_before']].values.tolist(),
                           same_point[['x_after', 'y_after']].values.tolist())
    # copy_shp(shp_path, res_path)
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
    # 为了使属性表字段支持中文，请添加下面这句
    gdal.SetConfigOption("SHAPE_ENCODING", "")
    res_name = os.path.basename(shp_path)
    strVectorFile = os.path.join(res_path,res_name)
    # 注册所有的驱动
    ogr.RegisterAll()
    # 创建数据，这里以创建ESRI的shp文件为例
    strDriverName = "ESRI Shapefile"
    oDriver = ogr.GetDriverByName(strDriverName)
    if oDriver == None:
        print("%s 驱动不可用！\n", strDriverName)
        return
    # 创建数据源
    oDS = oDriver.CreateDataSource(strVectorFile)
    if oDS == None:
        print("创建文件【%s】失败！", strVectorFile)
        return

    # 创建图层，创建一个多边形图层，这里没有指定空间参考，如果需要的话，需要在这里进行指定
    papszLCO = []
    oLayer = oDS.CreateLayer("road", None, ogr.wkbLineString, papszLCO)
    if oLayer == None:
        print("图层创建失败！\n")
        return

    # 下面创建属性表
    # 先创建一个叫FieldID的整型属性
    oLayer.CreateField(ogr.FieldDefn("FieldID", ogr.OFTInteger))
    # 再创建一个叫FeatureName的字符型属性，字符长度为50
    oLayer.CreateField(ogr.FieldDefn("FieldName", ogr.OFTString))
    oDefn = oLayer.GetLayerDefn()

    file = shapefile.Reader(shp_path)
    polygons = file.shapes()  # 一个shp由多个多边形组成
    for i,shp_i in enumerate(polygons):
        poly_list = []
        polygon_point = shp_i.points  # 第一个多边形的点集
        for j, point in enumerate(polygon_point):
            x,y = road_tran.transform([point[0], point[1]])
            poly_list.append('{} {}'.format(x,y))
        oFeature = ogr.Feature(oDefn)
        oFeature.SetField("FieldID", i)
        oFeature.SetField("FieldName", "test")
        geomLine = ogr.CreateGeometryFromWkt("LINESTRING ({})".format(",".join(poly_list)))
        oFeature.SetGeometry(geomLine)
        oLayer.CreateFeature(oFeature)
    file.close()
    oDS.Destroy()
    print("数据集创建完成！\n")



# def copy_shp(shp_path, res_path):
#     shp_name = os.path.basename(shp_path)
#     source_path = os.path.dirname(shp_path)
#     base_name, _ = os.path.splitext(shp_name)
#     ext_name = ['dbf', 'prj', 'shp', 'shx']
#     for ext in ext_name:
#         file_name = '{}.{}'.format(base_name, ext)
#         shutil.copyfile(os.path.join(source_path, file_name), os.path.join(res_path, file_name))


def test():
    from_pt = ((38671803.6437, 2578831.9242), (38407102.8445, 2504239.2774), (38122268.3963, 2358570.38514),
               (38126455.4595, 2346827.2602), (38177232.2601, 2398763.77833), (38423567.3485, 2571733.9203),
               (38636876.4495, 2543442.3694), (38754169.8762, 2662401.86536), (38410773.8815, 2558886.6518),
               (38668962.0430, 2578747.6349))  # 输入点坐标对
    to_pt = ((38671804.6165, 2578831.1944), (38407104.0875, 2504239.1898), (38122269.2925, 2358571.57626),
             (38126456.5675, 2346826.27022), (38177232.3973, 2398762.11714), (38423565.7744, 2571735.2278),
             (38636873.6217, 2543440.7216), (38754168.8662, 2662401.86101), (38410774.5621, 2558886.0921),
             (38668962.5493, 2578746.94))  # 输出点坐标对

    trn = affine_fit(from_pt, to_pt)

    if trn:
        print("转换公式:")
        print(trn.To_Str())
        err = 0.0
        for i in range(len(from_pt)):
            fp = from_pt[i]
            tp = to_pt[i]
            t = trn.transform(fp)
            print("%s => %s ~= %s" % (fp, tuple(t), tp))
            err += ((tp[0] - t[0]) ** 2 + (tp[1] - t[1]) ** 2) ** 0.5

        print("拟合误差 = %f" % err)

if __name__ == "__main__":
    test()




