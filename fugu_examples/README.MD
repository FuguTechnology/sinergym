# 数据中心模拟环境示例

注:因为能耗返回的需求率和之前anyhvac的计算方式不太一样，奖励函数目前是在sinergym自带的ExpReward的基础上去掉了温度过低惩罚，调整了一下能耗和温度惩罚的系数。
实际验证时参数可以进行调整。
 
## 单热区小数据中心基本参数
    
    外部环境:示例模型中设定使用深圳天气 
    
    数据中心尺寸： 15.24 * 15.24 * 4.572 
    
    风冷空调主要参数:COP=4.5669，额定制冷量148kw，额定风量8.5m³/s，风扇电机总效率0.7 
    
    IT设备主要参数：100台服务器，每台功率500W， 电源效率0.9
    
    动作空间:制冷回风温度设定值（16,32）
    
    模拟步长：1分钟
    
    模拟时限：1年
    
    观测空间包含以下值：
    
        month: 月份
        day_of_month: 日期（当月的第几天）
        hour: 小时
        outdoor_temperature: 室外空气干球温度
        outdoor_humidity: 室外空气相对湿度
        wind_speed: 风速
        wind_direction: 风向
        diffuse_solar_radiation: 每单位面积的漫射太阳辐射率
        direct_solar_radiation: 每单位面积的直射太阳辐射率
        htg_setpoint: 区域恒温器供暖设定温度  
        clg_setpoint: 区域恒温器制冷设定温度
        air_temperature: 区域空气温度
        air_humidity: 区域空气相对湿度
        cooling_coil_demand_rate: 冷却盘管电力需求率
        fan_demand_rate: 风机电力需求率
        ups_demand_rate: IT设备UPS电力需求率
        ite_fan_demand_rate: IT设备风机电力需求率
        cpu_demand_rate: IT设备CPU电力需求率
        HVAC_electricity_demand_rate: 建筑HVAC总电力需求率
        building_electricity_demand_rate: 建筑总电力需求率
        total_electricity_demand_rate: 建筑总电力需求率
    
    
    example运行方式
    先根据项目install文档安装docker环境，然后运行的时候挂载修改后的代码到项目/workspaces/sinergym目录下

    docker run --mount type=bind,source=$yourSinergymPath,target=/workspaces/sinergym -it --rm sailugr/sinergym /bin/bash
    python fugu_examples/small_datacenter_examply_v1.py

    
    v1版本修改
    
    1.修改模拟步长 
    
    2.动作空间去掉了风机设定值，只保留了温度设定（实测没什么效果） 
    
    3.修改设备功率和比热容调试 
    
    4.奖励方程只考虑温度过高，调整计算参数,添加偏置值 
    
    5.使用深圳天气环境

## 4热区4空调大数据中心模拟环境参数

    外部环境:同样使用深圳天气 1分钟控制步长 
    
    数据中心尺寸： 约20 * 10 * 5 m³,分为4个热区，每个热区配有一个风冷精密末端空调系统
    4个空间尺寸分别大概为: 
    THERMAL ZONE 1  11.6 * 6.8 * 5 m³
    THERMAL ZONE 2  9.2 * 6.8 * 5 m³
    THERMAL ZONE 3  11.6 * 4.5 * 5 m³
    THERMAL ZONE 4  9.2 * 4.5 * 5 m³
    
    风冷空调额定功率根据4个区域大小和区域负载分配，分别为（估算值，可能与模式实际值有少量偏差）：
    空间 | 面积 (m²) | 额定制冷量 (kW) | 额定风量 (m³/min)
    THERMAL ZONE 1 | 79.09 | 46.66 | 153.96
    THERMAL ZONE 2 | 62.96 | 37.15 | 122.58
    THERMAL ZONE 3 | 51.78 | 30.55 | 100.80
    THERMAL ZONE 4 | 41.22 | 24.32 | 80.28

    
    IT设备主要参数：设备功率约500W每平方米，灯光约9.6W每立方米
    
    动作空间:4区域制冷回风温度设定gym.spaces.Box(low=np.array([16.0, 16.0,16.0,16.0], dtype=np.float32), high=np.array([32.0, 32.0,32.0, 32.0], dtype=np.float32), shape=(4,), dtype=np.float32)
    
    模拟步长：1分钟
    
    模拟时限：1年
    
    观测空间包含以下值：
    
        month: 月份
        day_of_month: 日期（当月的第几天）
        hour: 小时
        outdoor_temperature: 室外空气干球温度
        outdoor_humidity: 室外空气相对湿度
        wind_speed: 风速
        wind_direction: 风向
        thermal_zone_1_air_temperature: 区域1空气温度
        thermal_zone_2_air_temperature: 区域2空气温度
        thermal_zone_3_air_temperature: 区域3空气温度
        thermal_zone_4_air_temperature: 区域4空气温度
        space_101_electric_equipment_5_electricity_equipment_energy: 区域1其他设备电力消耗
        space_102_electric_equipment_5_electricity_equipment_energy: 区域2其他设备电力消耗
        space_103_electric_equipment_5_electricity_equipment_energy: 区域3其他设备电力消耗
        space_104_electric_equipment_5_electricity_equipment_energy: 区域4其他设备电力消耗
        HVAC_electricity_demand_rate: 建筑HVAC总电力需求率

    
    
    example运行方式
    先根据项目install文档安装docker环境，然后运行的时候挂载修改后的代码到项目/workspaces/sinergym目录下

    docker run --mount type=bind,source=$yourSinergymPath,target=/workspaces/sinergym -it --rm sailugr/sinergym /bin/bash
    python fugu_examples/4zone_datacenter_example.py

