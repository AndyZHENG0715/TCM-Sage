$(function () {
    initTable();
});

$(document).ready(function(){
    $("#individual").hide();
    $("#enrichment_result").hide();
    $("#enrichment_svg").hide();
    $("#dl-btn").click(function(){
        do_dl();
        return false;
    });
    $("#dl_img").click(function(){
        dl_img('#cy');
        return false;
    });
    $("#dl_img_mol").click(function(){
        dl_img('#mol_cy');
        return false;
    });
    $("#dl_img_sym").click(function(){
        dl_img('#sym_cy');
        return false;
    });
    $("#rst_img").click(function(){
        cy.reset();
        return false;
    });
    $("#rst_img_mol").click(function(){
        mol_cy.reset();
        return false;
    });
    $("#rst_img_sym").click(function(){
        sym_cy.reset();
        return false;
    });
    $('#layout_select').change(function(){
        var layout_type = $("#layout_select").val();
        switch(layout_type)
        {
            case 'symmap':
                cy.layout({'name':'symmap_layout', 'animate': true}).run()
                cy.reset();
                break;
            case 'circle':
                cy.layout({'name':'circle', 'animate': true}).run()
                break;
            case 'concentric':
                cy.layout({'name':'concentric', 'animate': true}).run()
                break;
            case 'grid':
                cy.layout({'name':'grid', 'animate': true}).run()
                break;
        }
    });
    $('#layout_select_mol').change(function(){
        var layout_type = $("#layout_select_mol").val();
        switch(layout_type)
        {
            case 'Ingredient':
                mol_cy.layout({'name':'mol_layout', 'animate': true}).run()
                mol_cy.reset();
                break;
            case 'circle':
                mol_cy.layout({'name':'circle', 'animate': true}).run()
                break;
            case 'concentric':
                mol_cy.layout({'name':'concentric', 'animate': true}).run()
                break;
            case 'grid':
                mol_cy.layout({'name':'grid', 'animate': true}).run()
                break;
        }
    });
    $('#layout_select_sym').change(function(){
        var layout_type = $("#layout_select_sym").val();
        switch(layout_type)
        {
            case 'Herb-syndrome-symptom':
                sym_cy.layout({'name':'chsym_layout', 'animate': true}).run()
                sym_cy.reset();
                break;
            case 'circle':
                sym_cy.layout({'name':'circle', 'animate': true}).run()
                break;
            case 'concentric':
                sym_cy.layout({'name':'concentric', 'animate': true}).run()
                break;
            case 'grid':
                sym_cy.layout({'name':'grid', 'animate': true}).run()
                break;
        }
    });
    $(document).on('click', '#button_select_group button', function() {
        var content = $(this).text();
        change_browse_flag = true;
        $('#button_select_group button').each(function (){
            if($(this).hasClass("layui-btn-primary")){
            }else {
                $(this).addClass("layui-btn-primary");
            }
        })
        $(this).removeClass("layui-btn-primary");
        filter_text = filter_dic[content];
        change_browse();
    });

    $('#browse_select').change(function(){
    　　change_browse();
    });
    $('#type_select').change(function(){
        var type_val = $('#type_select').val();
        switch (type_val) {
            case '0':
                $('#table').bootstrapTable('filterBy',{}, {
                    'filterAlgorithm': function (row, filters){
                        return true;
                    }
                });
                break;
            case '1':
                $('#table').bootstrapTable('filterBy',{type: ['QC ingredients']}, {
                    'filterAlgorithm': function(row, filters){
                        var tmp = row.type
                        if(tmp.search(filters.type[0]) !== -1) return true;
                        return false;
                    }
                });
                break;
            case '2':
                $('#table').bootstrapTable('filterBy',{type: ['Blood ingredients']}, {
                    'filterAlgorithm': function(row, filters){
                        var tmp = row.type
                        if(tmp.search(filters.type[0]) !== -1) return true;
                        return false;
                    }
                });
                break;
            case '3':
                $('#table').bootstrapTable('filterBy',{type: ['Metabolic ingredients']}, {
                    'filterAlgorithm': function(row, filters){
                        var tmp = row.type
                        if(tmp.search(filters.type[0]) !== -1) return true;
                        return false;
                    }
                });
                break;
        }
    });
    $('#browse_sort').change(function(){
        var browse_sort = $("#browse_sort").val();
        switch(browse_sort)
        {
            case '0':
                $('#table').bootstrapTable('refreshOptions',{sortName:''});
                break;
            case '1':
                $('#table').bootstrapTable('refreshOptions',{sortName:'P_value'});
                break;
            case '2':
                $('#table').bootstrapTable('refreshOptions',{sortName:'FDR(BH)'});
                break;
            case '3':
                $('#table').bootstrapTable('refreshOptions',{sortName:'FDR(Bonferroni)'});
                break;
        }
    });
    change_browse();

    $('#ex1').slider({
	    formatter: function(value) {
	        cy.zoom( value );
	        return 'Current value: ' + value;
	    }
    });
    $('#ex2').slider({
	    formatter: function(value) {
	        mol_cy.zoom( value );
	        return 'Current value: ' + value;
	    }
    });
    $('#ex3').slider({
	    formatter: function(value) {
	        sym_cy.zoom( value );
	        return 'Current value: ' + value;
	    }
    });
});

var table_name_re;
// console.log(table_name_re)
var filter_text = (table_name_re === 'Herb')?'TCM_symptom': 'Herb';
// console.log(filter_text)
var filter_dic = {'Herb':'Herb', 'TCM Symptom':'TCM_symptom', 'MM Symptom':'MM_symptom', 'Ingredient':'Mol', 'Target': 'Gene', 'Disease': 'Disease', 'Syndrome': 'Syndrome'};
function get_relation(rel){
    table_name_re = rel;
    filter_text = (table_name_re === 'Herb')?'TCM_symptom': 'Herb';
    return rel;
}

function initTable(){
    // $("#table").colResizable({
    //     gripInnerHtml: "<div class='grip'></div>",
    //     postbackSafe: true, //刷新后保留之前的拖拽宽度
    // });
    $('#table').bootstrapTable({
        height: "auto",
        toolbar: '#toolbar',
        pagination:true,
        cache: false,
        sidePagination:'client',
        pageSize:10,
        pageList:[10,20,30,50],
        data: [],
        columns: [],
        onSort: function(name, order){
            switch(name)
            {
                case 'P_value':
                    $("#browse_sort").val('1');
                    break;
                case 'FDR(BH)':
                    $("#browse_sort").val('2');
                    break;
                case 'FDR(Bonferroni)':
                    $("#browse_sort").val('3');
                    break;
            }
            $("#browse_sort").selectpicker('refresh');
        }
    })
    $('#pathway_table').bootstrapTable({
        height: "auto",
        toolbar: '#toolbar',
        pagination:true,
        cache: false,
        sidePagination:'client',
        pageSize:5,
        pageList:[5,10,20,30,50],
        data: [],
        columns: []
    });
    $('#go_bp_table').bootstrapTable({
        height: "auto",
        toolbar: '#toolbar',
        pagination:true,
        cache: false,
        sidePagination:'client',
        pageSize:5,
        pageList:[5,10,20,30,50],
        data: [],
        columns: []
    });
    $('#go_cc_table').bootstrapTable({
        height: "auto",
        toolbar: '#toolbar',
        pagination:true,
        cache: false,
        sidePagination:'client',
        pageSize:5,
        pageList:[5,10,20,30,50],
        data: [],
        columns: []

    });
    $('#go_mf_table').bootstrapTable({
        height: "auto",
        toolbar: '#toolbar',
        pagination:true,
        cache: false,
        sidePagination:'client',
        pageSize:5,
        pageList:[5,10,20,30,50],
        data: [],
        columns: []
    });
};
function do_dl(){
    //创建form表单
    var params = new Array();
    var browse_type = filter_text;
    var source_type = $("#req_name").val();
    var filter_type = $('#browse_select').val();

    params.push({ name: "rrid", value: source_type});
    params.push({ name: "table_name", value: browse_type});
    params.push({ name: "filter", value: filter_type});
    console.log(params)

    var temp_form = document.createElement("form");
    temp_form.action = "/dl/";
    //如需打开新窗口，form的target属性要设置为'_blank'
    temp_form.target = "_self";
    temp_form.method = "post";
    temp_form.style.display = "none";
    //添加参数
    for (var item in params) {
        var opt = document.createElement("textarea");
        opt.name = params[item].name;
        opt.value = params[item].value;
        temp_form.appendChild(opt);
    }
    document.body.appendChild(temp_form);
    //提交数据
    temp_form.submit();
}

function dl_img(id){
    html2canvas(document.querySelector(id)).then(canvas => {
        // 图片导出为 png 格式
        var type = 'png';
        var imgData = canvas.toDataURL(type);
        var _fixType = function(type) {
            type = type.toLowerCase().replace(/jpg/i, 'jpeg');
            var r = type.match(/png|jpeg|bmp|gif/)[0];
            return 'image/' + r;
        };

        // 加工image data，替换mime type
        imgData = imgData.replace(_fixType(type),'image/octet-stream');


        var saveFile = function(data, filename){
            var save_link = document.createElementNS('http://www.w3.org/1999/xhtml', 'a');
            save_link.href = data;
            save_link.download = filename;

            var event = document.createEvent('MouseEvents');
            event.initMouseEvent('click', true, false, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
            save_link.dispatchEvent(event);
        };

        // 下载后的文件名
                var filename = 'network'+ '.' + type;
        // download
                saveFile(imgData,filename);

    });
}


function change_browse(){
    var browse_type = filter_text;
    var source_type = $("#req_name").val();
    var filter_type = $('#browse_select').val();
    var rl_dr = {
        'Herb': ['Mol', 'TCM_symptom', 'MM_symptom', 'Syndrome'],
        'Mol': ['Herb', 'Gene'],
        'TCM_symptom': ['Herb', 'MM_symptom', 'Syndrome'],
        'MM_symptom': ['TCM_symptom', 'Disease'],
        'Gene': ['Mol', 'Disease'],
        'Disease':['MM_symptom', 'Gene'],
        'Syndrome':['Herb', 'TCM_symptom']
    };

    var conv_name= {
        'SMHB': 'Herb',
        'SMIT': 'Mol',
        'SMTS': 'TCM_symptom',
        'SMMS': 'MM_symptom',
        'SMTT': 'Gene',
        'SMDE': 'Disease',
        'SMSY': 'Syndrome'
    };

    if (rl_dr[conv_name[source_type.substr(0, 4)]].indexOf(browse_type) >= 0){
        $("#browse_select").val('0');
        $("#browse_sort").val('0');
        $(".ind_sl").attr("disabled",true);
        $("#browse_select").selectpicker('refresh');
        $("#browse_sort").selectpicker('refresh');
    }
    else{
        $(".ind_sl").attr("disabled",false);
        $("#browse_select").selectpicker('refresh');
        $("#browse_sort").selectpicker('refresh');
    }

    if(browse_type === 'Gene'){
        $("#enrichment-btn").css("display", "block");

        // $('#enrichment_result').show();
        $('#table_download').show();
        $("#enrichment_svg").show();
    }
    else {
        $("#enrichment-btn").css("display", "none");
        $("#individual").hide();
        // $('#enrichment_result').hide();
        $('#table_download').hide();
        $("#enrichment_svg").hide();
    }
    if(browse_type === 'Mol' && table_name_re === 'Herb'){
        $("#footer_ingredient").show();
        $("#ingredients_type").show();
        $("#footer_tcm_symptom").hide();
        $("#footer_syndrome").hide();
    }
    else if(browse_type === 'TCM_symptom' && table_name_re === 'Herb'){
        $("#footer_tcm_symptom").show();
        $("#footer_ingredient").hide();
        $("#ingredients_type").hide();
        $("#footer_syndrome").hide();
    }
    else if(browse_type === 'Syndrome'){
        $("#footer_syndrome").show();
        $("#footer_ingredient").hide();
        $("#ingredients_type").hide();
        $("#footer_tcm_symptom").hide();
    }
    else {
        $("#footer_ingredient").hide();
        $("#ingredients_type").hide();
        $("#footer_tcm_symptom").hide();
        $("#footer_syndrome").hide();
    }
    $('#table').bootstrapTable("refreshOptions", {columns: [], data: [], formatNoMatches:function(){return "Searching, please wait...";}});
    $.ajax({
        url:"/related_components/",
        type:"post",
        data:{
            rrid : source_type,
            table_name : browse_type,
            filter : filter_type,
        },
        success:function(res){

            var data = $.parseJSON(res);
            if (data['data'].length == 0){
                $('#table').bootstrapTable("refreshOptions", {columns: [], data: [], formatNoMatches:function(){return "No result found.";}});
                if(browse_type === 'Gene'){
                    $("#individual").hide();
                }
            }
            else{
                if(browse_type !== 'Gene'){
                    data.columns[0].formatter = aFormatter;
                    $('#table').bootstrapTable("refreshOptions", {columns: data['columns'], data: data['data']});
                    var tdiv = "<div class='floating' style='background: white; border: 1px solid black; POSITION: ABSOLUTE; margin-left: 15%;color:black;'></div>"

                    $('[data-field="Value"]').append(tdiv);
                    $(".floating").hide();
                    $('[data-field="Value"]').hover(function (){
                        $(".floating").show();

                        $(".floating").html('Inferred evidence score (IES) is a reliability score that measure inferred relationship based on connection evidence of heterogeneous network. The higher the score, the more reliable it is.<a href="/help/">See details in here.</a>');

                    },
                        function (){
                         $(".floating").hide();
                        }
                    );


                }
                else {
                    $("#individual").show();
                    data['columns'].unshift({checkbox:true});
                    data.columns[1].formatter = aFormatter;
                    $('#table').bootstrapTable("refreshOptions", {columns: data['columns'], data: data['data'],
                        onCheck: function (row){
                            $('#geneText').tagEditor('addTag', row['Gene_symbol']);
                        },
                        onUncheck: function (row){
                            $('#geneText').tagEditor('removeTag', row['Gene_symbol']);
                        },
                        onCheckAll:function (rows){
                            for(var i=0; i<rows.length; i++){
                                $('#geneText').tagEditor('addTag', rows[i]['Gene_symbol']);
                            }
                        },
                        onUncheckAll:function (rows){
                            for (var i=0; i<rows.length; i++){
                                $('#geneText').tagEditor('removeTag', rows[i]['Gene_symbol']);
                            }
                        }
                    });
                    var geneList = []
                    for(var item in data['data']){
                        geneList.push(data['data'][item]['Gene_symbol']);
                    }
                    sessionStorage.setItem('geneList', geneList);
                    enrichment_GO(geneList)
                }
                $('#table').removeClass("table-bordered");
                $('#table').removeClass("table-striped");



            }
        },
        error:function(e){
            location.reload(true);
            //alert("The token has expired, please refresh the page.");
        },
    });
}

//添加超链接
function aFormatter(value, row, index) {
    return [
        '<a href="/detail/'+value+'">'+value+'</a>'
    ].join("")
};

function enrichmentGo(){
    var gene_list = [];
    var rows = $('#table').bootstrapTable('getSelections');

    if (rows.length === 0) {
        layer.alert("Please select at least one gene.");
        return;
    } else {
        $(rows).each(function () {
            gene_list.push(this.Gene_symbol);
        });
    }
    if($("#geneText").css("display") === 'none'){
        $("#geneText").css("display", "block");
        $("#geneText").tagEditor({initialTags: gene_list, forceLowercase: false, clickDelete: true,
            // beforeTagDelete: function(field, editor, tags, val) {
            //     $('#table').bootstrapTable('uncheckBy', {field:'Gene_symbol', values: [val]});
            //     return true;
            // }
        });
        $("#enrichment-btn").addClass("disabled");
        $("#sub-btn").css("display", "block");
        $("#res-btn").css("display", "block");
    }
}

function insubmit(){
    var geneList = $("#geneText").tagEditor('getTags')[0].tags;
    enrichment_GO(geneList)
}

function enrichment_GO(geneList = []){

    //更新curated症状基因--表格数据
    $('#pathway_table').bootstrapTable("refreshOptions", {columns: [], data: [], formatNoMatches:function(){return "Searching, please wait...";}});
    $('#go_bp_table').bootstrapTable("refreshOptions", {columns: [], data: [], formatNoMatches:function(){return "Searching, please wait...";}});
    $('#go_cc_table').bootstrapTable("refreshOptions", {columns: [], data: [], formatNoMatches:function(){return "Searching, please wait...";}});
    $('#go_mf_table').bootstrapTable("refreshOptions", {columns: [], data: [], formatNoMatches:function(){return "Searching, please wait...";}});
    $.ajax({
        url:"/related_enrichment/",
        type:"post",
        data:{
            genes:JSON.stringify({'gene':geneList})
        },
        success:function(res){
            var data = $.parseJSON(res);
            data_enrich=data['enrichment'];
            data_GO_bp=data['GO_bp'];
            data_GO_cc=data['GO_cc'];
            data_GO_mf=data['GO_mf'];

            //lucas 190618更新enrichment表数据
            if (data_enrich['data'].length === 0){
                // $('#pathway_table').bootstrapTable("refreshOptions", {columns: [], data: [], formatNoMatches:function(){return "No result found.";}});
                InitChart('pathway_svg', data_enrich['data'])
            }
            else{
                // data_enrich.columns[0].formatter = aFormatter_pathway;
                // $('#pathway_table').bootstrapTable("refreshOptions", {columns: data_enrich['columns'], data: data_enrich['data']});
                InitChart('pathway_svg', data_enrich['data'])
                window.sessionStorage.setItem("pathway_table", JSON.stringify({"columns": data_enrich['columns'], "data": data_enrich['data']}));
            }
             //lucas 190618更新GO-BP表数据
            if (data_GO_bp['data'].length === 0){
                // $('#go_bp_table').bootstrapTable("refreshOptions", {columns: [], data: [], formatNoMatches:function(){return "No result found.";}});
            }
            else{
                // data_GO_bp.columns[0].formatter = aFormatter_GO
                // $('#go_bp_table').bootstrapTable("refreshOptions", {columns: data_GO_bp['columns'], data: data_GO_bp['data']});
                InitChart('go_bp_svg', data_GO_bp['data'])
                window.sessionStorage.setItem("go_bp_table", JSON.stringify({"columns": data_GO_bp['columns'], "data": data_GO_bp['data']}));
            }
            //lucas 190618更新GO-CC表数据
            if (data_GO_cc['data'].length === 0){
                // $('#go_cc_table').bootstrapTable("refreshOptions", {columns: [], data: [], formatNoMatches:function(){return "No result found.";}});
            }
            else{
                // data_GO_cc.columns[0].formatter = aFormatter_GO
                // $('#go_cc_table').bootstrapTable("refreshOptions", {columns: data_GO_cc['columns'], data: data_GO_cc['data']});
                InitChart('go_cc_svg', data_GO_cc['data'])
                window.sessionStorage.setItem("go_cc_table", JSON.stringify({"columns": data_GO_cc['columns'], "data": data_GO_cc['data']}));
            }
             //lucas 190618更新GO-MF表数据
            if (data_GO_mf['data'].length === 0){
                // $('#go_mf_table').bootstrapTable("refreshOptions", {columns: [], data: [], formatNoMatches:function(){return "No result found.";}});
            }
            else{
                // data_GO_mf.columns[0].formatter = aFormatter_GO
                // $('#go_mf_table').bootstrapTable("refreshOptions", {columns: data_GO_mf['columns'], data: data_GO_mf['data']});
                InitChart('go_mf_svg', data_GO_mf['data'])
                window.sessionStorage.setItem("go_mf_table", JSON.stringify({"columns": data_GO_mf['columns'], "data": data_GO_mf['data']}));
            }
        },
        error:function(e){
            location.reload(true);
            //alert("The token has expired, please refresh the page.");
        }
    });
}

function enrichReset(){
    var tags = $('#geneText').tagEditor('getTags')[0].tags;
    for (i = 0; i < tags.length; i++) { $('#geneText').tagEditor('removeTag', tags[i]); }
    $('#geneText').tagEditor('destroy');
    $("#geneText").css("display", "none");
    $("#sub-btn").css("display", "none");
    $("#res-btn").css("display", "none");
    $('#table').bootstrapTable('uncheckAll');
    $("#enrichment-btn").removeClass("disabled");
    var geneList = sessionStorage.getItem('geneList');
    enrichment_GO(geneList.split(','));
}
//添加GO超链接
function aFormatter_GO(value, row, index) {
    return [
        '<a href="http://amigo.geneontology.org/amigo/term/'+value+'">'+value+'</a>'
    ].join("");
}

//添加pathway超链接
function aFormatter_pathway(value, row, index) {
    return [
        '<a href="https://www.kegg.jp/kegg-bin/show_pathway?'+value+'">'+value+'</a>'
    ].join("");
}

function InitChart(div_name, enrich_data){
    var xArray = [];
    var yArray = [];
    var zArray = [];
    var sizeArray = [];
    var show_length = 0;
    var title = '';
    if(enrich_data.length <= 20){
        show_length = enrich_data.length
    }else{
        show_length = 20
    }
    if(show_length !== 0){
        ylabel_length = 0
        for(var item = 0; item<show_length; item++){
            if(enrich_data[item][2] !== '>1000'){
                xArray.push(enrich_data[item][2])
                if(enrich_data[item][1].length >= 100){
                    var yArray_up = ''
                    var enter_flag = true
                    var str_tmp = enrich_data[item][1].split(' ')
                    for(var i in str_tmp){
                        yArray_up += str_tmp[i] + ' '
                        if(yArray_up.length >= 90 && enter_flag){
                            yArray_up += '<br>'
                            enter_flag = false
                        }
                    }
                    yArray.push(yArray_up)
                }
                else{
                    yArray.push(enrich_data[item][1] + ' ')
                }
                if(enrich_data[item][1].length >= ylabel_length) ylabel_length = enrich_data[item][1].length
                zArray.push(-Math.log(enrich_data[item][3]))
                sizeArray.push(enrich_data[item][5]*3)
            }
        }
        if(ylabel_length >= 100){
            ylabel_length = 100
        }
        if(xArray.length === 0){
            xArray.push("Infinity, please download the enrichment results to view details")
            yArray.push(0)
            zArray.push(0)
            sizeArray.push(0)
        }
    }
    else {
        xArray.push("There are no results related to these genes")
        yArray.push(0)
        zArray.push(0)
        sizeArray.push(0)
    }
    switch(div_name) {
        case 'pathway_svg':
            title = 'Pathway enrichment'
            break;
        case 'go_bp_svg':
            title = 'GO (biological process) enrichment'
            break;
        case 'go_cc_svg':
            title = 'GO (cellular component) enrichment'
            break;
        case 'go_mf_svg':
            title = 'GO (molecular function) enrichment'
            break;
    }

    var traceA = {
        type: "scatter",
        mode: "markers",
        x: xArray,
        y: yArray,
        marker: {
            color: zArray,
            colorscale: [[0, 'rgb(78,52,46)'], [0.5, 'rgb(245,127,23)'], [0.75, 'rgb(130,0,20)']],
            cmin: Math.min(zArray),
            cmax: Math.max(zArray),
            size: sizeArray,
            sizeref: 0.2,
            sizemode: 'area',
            showscale: true,
            colorbar: {
                thickness: 15,
                y: 1,
                yanchor: 'top',
                ypad: 4,
                len: 1,
                title: '-log10(P-value)',
                titleside: 'bottom',
                outlinewidth: 0,
                tickfont: {
                    family: 'Arial, sans-serif',
                    size: 10,
                    color: 'green'
                }
            },
        },

    };

    var data = [traceA];

    var layout = {
        title: {
            text:title,
            font:{
                family: 'Arial, sans-serif',
            },
        },

        margin:{t:50, r:100, b:70, l:ylabel_length*6.6},
        showlegend: false,
        width: 1100,
        height: 650,
        backgroundcolor: '#393636',
        xaxis:{
            title:"Gene ratio",
            rangemode: 'tozero',
        },
    plot_bgcolor: '#f5f5f5'
    };
    var config = {
      toImageButtonOptions: {
        format: 'svg',
        filename: title,
        height: 650,
        width: 1100,
        scale: 1
      },
      displayModeBar: true
    };
    Plotly.newPlot(div_name, data, layout, config)
}

function download_enrichment_result(){
    var sheetName = ['pathway', 'GO-BP', 'GO-CC', 'GO-MF']
    var sessionList = ['pathway_table', "go_bp_table", "go_cc_table", "go_mf_table"]
    var sheetList = []
    for(var i in sessionList){
        var aoa = $.parseJSON(sessionStorage.getItem(sessionList[i]))
        aoa['data'].unshift(aoa['columns'])
        var sheet = XLSX.utils.aoa_to_sheet(aoa['data'])
        sheetList.push(sheet)
    }

    var bob = sheet2blob(sheetList, sheetName)
    openDownloadDialog(bob, 'enrichment_result.xlsx')
}
function sheet2blob(sheet, sheetName) {
	sheetName = sheetName || 'sheet1';
	var workbook = {
		SheetNames: sheetName,
		Sheets: {}
	};
	for(var i in sheetName){
	    workbook.Sheets[sheetName[i]] = sheet[i];
    }

	// 生成excel的配置项
	var wopts = {
		bookType: 'xlsx', // 要生成的文件类型
		bookSST: false, // 是否生成Shared String Table，官方解释是，如果开启生成速度会下降，但在低版本IOS设备上有更好的兼容性
		type: 'binary'
	};
	var wbout = XLSX.write(workbook, wopts);
	var blob = new Blob([s2ab(wbout)], {type:"application/octet-stream"});
	// 字符串转ArrayBuffer
	function s2ab(s) {
		var buf = new ArrayBuffer(s.length);
		var view = new Uint8Array(buf);
		for (var i=0; i!=s.length; ++i) view[i] = s.charCodeAt(i) & 0xFF;
		return buf;
	}
	return blob;
}
function openDownloadDialog(url, saveName) {
	if(typeof url == 'object' && url instanceof Blob)
	{
		url = URL.createObjectURL(url); // 创建blob地址
	}
	var aLink = document.createElement('a');
	aLink.href = url;
	aLink.download = saveName || ''; // HTML5新增的属性，指定保存文件名，可以不要后缀，注意，file:///模式下不会生效
	var event;
	if(window.MouseEvent) event = new MouseEvent('click');
	else
	{
		event = document.createEvent('MouseEvents');
		event.initMouseEvent('click', true, false, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
	}
	aLink.dispatchEvent(event);
}

function showResult(obj){
    var text = $(obj).val()
    var box = $(obj).prev()
    if(text === 'Fold'){
        box.removeClass("hidden")
        $(obj).val('Unfold')
        $(obj).html('Unfold<i class="glyphicon glyphicon-triangle-top"></i>')
    }
    else{
        box.addClass("hidden")
        $(obj).val('Fold')
        $(obj).html('Fold<i class="glyphicon glyphicon-triangle-bottom"></i>')
    }

}


function showShortResult(obj){
    var text = $(obj).val()
    var longbox = $(obj).prev()
    var shortBox = $(obj).prev().prev()
    if(text === 'Fold'){
        longbox.removeClass("hidden")
        shortBox.addClass("hidden")
        $(obj).val('Unfold')
        $(obj).html('Unfold<i class="glyphicon glyphicon-triangle-top">')
    }
    else{
        longbox.addClass("hidden")
        shortBox.removeClass("hidden")
        $(obj).val('Fold')
        $(obj).html('Fold<i class="glyphicon glyphicon-triangle-bottom">')
    }

}