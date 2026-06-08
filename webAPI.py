import json
from flask import Flask, request, jsonify
from pipeNetworkSimulation import PipeNetworkSimulation
from Fluid.componentsDef import Phaseplot


app = Flask(__name__)

@app.route('/pipeNetworkSim', methods=['POST'])
def pipe_network_sim_api():

    try:
        request_data = request.get_json()
        file_or_json = request_data.get("fileOrJson")
        excel_path = request_data.get("excelPath", "./DB/NetworkProject-threeNodes.xlsx")
        json_network_model = request_data.get("jsonNetworkModel")
        is_save_picture = request_data.get("isSavePicture", False)
        is_save_result = request_data.get("isSaveResult", False)
        save_path = request_data.get("savePath", "./segGraphResult")
        grid_length = float(request_data.get("gridLength", 100.0))
        sink_p = request_data.get("sink_p", request_data.get("sinkP"))

        network_sim = PipeNetworkSimulation()
        raw_result = network_sim.start(
            fileOrJson=file_or_json,
            jsonNetworkModel=json_network_model,
            excelPath=excel_path,
            isSavePicture=is_save_picture,
            isSaveResult=is_save_result,
            savePath=save_path,
            gridLength=grid_length,
            sink_p=sink_p
        )

        # result = json.loads(raw_result) if raw_result else {}
        result=raw_result
        return jsonify({
            "status": "success",
            "code": 200,
            "message": "管网仿真计算完成",
            "result": result
        }), 200
    except Exception as  e:
        # 捕获所有异常，把错误信息放进 message
        return jsonify({
            "status": "error",
            "code": 500,
            "message": f"计算失败：{str(e)}",  # 异常信息在这里
            "result": None
        }), 200



@app.route('/Phaseplot', methods=['POST'])
def Phaseplot_api():

    try:
###################################   初始化   ###################################
        request_data = request.get_json()
        jsonPhaseplotModel = request_data["jsonPhaseplotModel"]
        CompositionalSim = Phaseplot.CompositionalSimulation()
        fluidMixInfo = CompositionalSim.injsondataprecess(jsonPhaseplotModel)
        Equition_type = "SRK"  # "SRK" "PR"
        fluid = Phaseplot.fluid2P(fluidMixInfo, Equition_type)

###################################   相图绘制   ###################################
        '''
        PDmin 露点压力最小值
        PDmax 露点压力最大值
        TBmin 泡点温度最小值
        TBmax 泡点温度最大值
        PDstep 压力步长
        TBstep 温度步长
        '''
        PDmin=100000
        PDmax=7*1000000
        TBmin=273.15- 90
        TBmax=273.15+ 90
        PDstep=100000
        TBstep=10
        fluid.Phase_diagram(PDmin,PDmax,TBmin,TBmax,PDstep,TBstep)
        result = fluid.phase_json_data


        return jsonify({
            "status": "success",
            "code": 200,
            "message": "相图计算完成!",
            "result": result
        }), 200
    except Exception as e:
        # 捕获所有异常，把错误信息放进 message
        return jsonify({
            "status": "error",
            "code": 500,
            "message": f"计算失败：{str(e)}",  # 异常信息在这里
            "result": None
        }), 200



if __name__ == "__main__":
    app.run(
        threaded=True,
        host='0.0.0.0',
        port=6050,
        debug=True
    )
