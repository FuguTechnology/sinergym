## 单区域数据中心模拟环境示例
 
### 基本参数

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
    cooling_coil_demand_rate: 冷却盘管电力需求速率
    fan_demand_rate: 风机电力需求速率
    ups_demand_rate: IT设备UPS电力需求速率
    ite_fan_demand_rate: IT设备风机电力需求速率
    cpu_demand_rate: IT设备CPU电力需求速率
    HVAC_electricity_demand_rate: 建筑HVAC总电力需求速率
    building_electricity_demand_rate: 建筑总电力需求速率
    total_electricity_demand_rate: 建筑总电力需求速率


### example运行方式
先根据项目install文档安装docker环境，然后运行的时候挂载修改后的代码到项目/workspaces/sinergym目录下
```
docker run --mount type=bind,source=$yourSinergymPath,target=/workspaces/sinergym -it --rm sailugr/sinergym /bin/bash
python fugu_examples/small_datacenter_examply.py
```

### v1版本修改
1.修改模拟步长
2.动作空间去掉了风机设定值，只保留了温度设定（实测没什么效果）
3.修改设备功率和比热容调试
4.奖励方程只考虑温度过高，调整计算参数,添加偏置值