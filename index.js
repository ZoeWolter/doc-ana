// Specify margins and size of svg
var margins = {
    top: 50,
    right: 50,
    bottom: 50,
    left: 50
};

const width = 800; 
const height = 800;

var svg = d3.select("#plot").append("svg")
    .attr("width", width)
    .attr("height", height)
    .append("g")
    .attr("transform", "translate(" + margins.left + "," + margins.top + ")");


var legend = d3.select('#legend').append('svg')
    .attr('width', 300)
    .attr('height', height)
    .append("g")
    .attr("transform", "translate(0," + margins.top + ")");

var svg2 = d3.select("#plot2").append("svg")
    .attr("width", width)
    .attr("height", height)
    .append("g")
    .attr("transform", "translate(" + margins.left + "," + margins.top + ")");


var legend2 = d3.select('#legend2').append('svg')
    .attr('width', 300)
    .attr('height', height)
    .append("g")
    .attr("transform", "translate(0," + margins.top + ")");

//define scales
var xScale = d3.scaleLinear().range([50, width-100]);
var yScale = d3.scaleLinear().range([height-100, 50]);
var sizeScale = d3.scaleLinear().range([10,100]);
var colorScale = d3.scaleLinear().domain([0,0.5,1]).range(['#ca0020','gray', '#0571b0']);

//global variables
const num_posts_men = 14538;
const num_posts_women = 9278;
var sizeMax;

//load data and create visulaization
d3.dsv(";","data/topics.csv", function(d){
    return {topic_name : d.Name, topic_id : d.Topic_id, Count : +d.Count, Count_men : +d.Count_men, Count_women : +d.Count_women, x : +d.x, y : +d.y, Representation : d.Representation, Representative_Docs: d.Representative_Docs}
}).then(function(d){
    var data = d;
    data.sort((a,b) => b.Count - a.Count); //sort according to overall occurence
    //var data20 = data.slice(0,20); //take only top 20 topics
    console.log(data);

    // adjust domains of defined scales to min and max of data
    xScale.domain(d3.extent(data, d => d.x));
    yScale.domain(d3.extent(data, d => d.y));
    sizeMax = d3.max(data, d => d.Count);
    sizeScale.domain([0,sizeMax]);

    // calculate relative ocuurences of topics in AskMen/Women -> adjusted by total posts in AskMen/Women
    for (let i = 0; i < data.length; i++) {
        data[i].share_men = data[i].Count_men/num_posts_men ; 
        data[i].share_women = data[i].Count_women/num_posts_women;
        data[i].topic_share = data[i].share_men / (data[i].share_men + data[i].share_women)
    };
    

    // add topics
    svg.selectAll('.topic').data(data).enter().append('circle')
        .attr('class', 'topic')
        .attr('cx', d=> xScale(d.x))
        .attr('cy', d => yScale(d.y))
        .style('r', d => sizeScale(d.Count)/2)
        .attr('fill', d => colorScale(d.topic_share))
        .on("mouseover", function() {
            d3.select(this).style("cursor", "pointer");
        })
        // add tooltip
        .on("mouseenter",function(event, d){
            d3.select("#tooltip").style("left",(event.pageX+30)+"px").style("top",(event.pageY-20)+"px").style("display","block")
            .html(
                '<strong>' + d.topic_name + ' (id ' + d.topic_id + '): </strong> <br><br> - ' +
                d.Count + ' posts (' + Math.round(d.topic_share * 10000) / 100 + '% AskMen, ' +
                Math.round((1 - d.topic_share) * 10000) / 100 + '% AskWomen)' 
                + '<br> - Top 10 words: ' + d.Representation
                //+ '<br> - representaive post: ' + d.Representative_Docs.slice(0, 1000) + '...'
              );
        })
        .on("mousemove",function(event, d){
            d3.select("#tooltip").style("left",(event.pageX+30)+"px").style("top",(event.pageY-20)+"px");
        })
        .on("mouseout",function(){
            d3.select("#tooltip").style("display","none");
            d3.select(this).style("cursor", "default");
        });

    // add color legend
    var gradient = legend.append("defs")
        .append("linearGradient")
        .attr("id", "gradient")
        .attr("x1", "0%")
        .attr("y1", "0%")
        .attr("x2", "100%")
        .attr("y2", "0%");

    gradient.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", "#ca0020"); 

    gradient.append("stop")
        .attr("offset", "50%")
        .attr("stop-color", "gray");

    gradient.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", "#0571b0");

    legend.append('text')
        .attr('x',0)
        .attr('y',80)
        .style('text-anchor', 'start')
        .html('Topic is present in:');

    legend.append("rect")
        .attr("width", 300)
        .attr("height", 20)
        .attr('x',0)
        .attr('y',100)
        .style("fill", "url(#gradient)");

    legend.append('text')
        .attr('x',0)
        .attr('y',140)
        .style('text-anchor', 'start')
        .html('AskWomen');
    legend.append('text')
        .attr('x',150)
        .attr('y',140)
        .style('text-anchor', 'center')
        .html('both');
    legend.append('text')
        .attr('x',300)
        .attr('y',140)
        .style('text-anchor', 'end')
        .html('AskMen');

    legend.append('text')
        .attr('x',10)
        .attr('y',250)
        .style('text-anchor', 'start')
        .html('- size: overall occurence of topic');
    
    legend.append('text')
        .attr('x',10)
        .attr('y',300)
        .style('text-anchor', 'start')
        .html('- position: 2-dim topic embedding');

    
    /**
    // add topics
    svg.selectAll('.topic').data(data20).enter().append('text')
        .attr('class', 'topic')
        .attr('x', d=> xScale(d.x))
        .attr('y', d => yScale(d.y))
        .style('font-size', d => sizeScale(d.Count))
        .style('text-anchor', 'center')
        .attr('fill', d => colorScale(d.topic_share))
        .html(d => d.topic_name)
        .on("mouseover", function() {
            d3.select(this).style("cursor", "pointer");
        })
        // add tooltip
        .on("mouseenter",function(event, d){
            d3.select("#tooltip").style("left",(event.pageX+30)+"px").style("top",(event.pageY-20)+"px").style("display","block")
                .html(d.topic_name + ' (id ' + d.topic_id +  '): ' + d.Count + ' posts (' + Math.round(d.topic_share*10000)/100 + '% AskMen, ' + Math.round((1-d.topic_share)*10000)/100 +'% AskWomen)');
        })
        .on("mousemove",function(event, d){
            d3.select("#tooltip").style("left",(event.pageX+30)+"px").style("top",(event.pageY-20)+"px")
                .html(d.topic_name + ' (id ' + d.topic_id +  '): ' + d.Count + ' posts (' + Math.round(d.topic_share*10000)/100 + '% AskMen, ' + Math.round((1-d.topic_share)*10000)/100 +'% AskWomen)');
        })
        .on("mouseout",function(){
            d3.select("#tooltip").style("display","none");
            d3.select(this).style("cursor", "default");
        });
         */
});

//second viz with raw data as input to BERTopic
d3.dsv(";","data/topics_raw_data.csv", function(d){
    return {topic_name : d.Name, topic_id : d.Topic_id, Count : +d.Count, Count_men : +d.Count_men, Count_women : +d.Count_women, x : +d.x, y : +d.y, Representation : d.Representation, Representative_Docs: d.Representative_Docs}
}).then(function(d){
    var data = d;
    data.sort((a,b) => b.Count - a.Count); //sort according to overall occurence
    //var data20 = data.slice(0,20); //take only top 20 topics
    console.log(data);

    // adjust domains of defined scales to min and max of data
    xScale.domain(d3.extent(data, d => d.x));
    yScale.domain(d3.extent(data, d => d.y));
    let sizeMax2 = d3.max(data, d => d.Count);
    if (sizeMax2 > sizeMax) { //adjust only if max counts are higher to enable comparison
        sizeScale.domain([0,sizeMax2]);
    }


    // calculate relative ocuurences of topics in AskMen/Women -> adjusted by total posts in AskMen/Women
    for (let i = 0; i < data.length; i++) {
        data[i].share_men = data[i].Count_men/num_posts_men ; 
        data[i].share_women = data[i].Count_women/num_posts_women;
        data[i].topic_share = data[i].share_men / (data[i].share_men + data[i].share_women)
    };
    

    // add topics
    svg2.selectAll('.topic').data(data).enter().append('circle')
        .attr('class', 'topic')
        .attr('cx', d=> xScale(d.x))
        .attr('cy', d => yScale(d.y))
        .style('r', d => sizeScale(d.Count)/2)
        .attr('fill', d => colorScale(d.topic_share))
        .on("mouseover", function() {
            d3.select(this).style("cursor", "pointer");
        })
        // add tooltip
        .on("mouseenter",function(event, d){
            d3.select("#tooltip").style("left",(event.pageX+30)+"px").style("top",(event.pageY-20)+"px").style("display","block")
            .html(
                '<strong>' + d.topic_name + ' (id ' + d.topic_id + '): </strong> <br><br> - ' +
                d.Count + ' posts (' + Math.round(d.topic_share * 10000) / 100 + '% AskMen, ' +
                Math.round((1 - d.topic_share) * 10000) / 100 + '% AskWomen)' 
                + '<br> - Top 10 words: ' + d.Representation
                //+ '<br> - representaive post: ' + d.Representative_Docs.slice(0, 1000) + '...'
              );
        })
        .on("mousemove",function(event, d){
            d3.select("#tooltip").style("left",(event.pageX+30)+"px").style("top",(event.pageY-20)+"px");
        })
        .on("mouseout",function(){
            d3.select("#tooltip").style("display","none");
            d3.select(this).style("cursor", "default");
        });

    // add color legend
    var gradient = legend2.append("defs")
        .append("linearGradient")
        .attr("id", "gradient")
        .attr("x1", "0%")
        .attr("y1", "0%")
        .attr("x2", "100%")
        .attr("y2", "0%");

    gradient.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", "#ca0020"); 

    gradient.append("stop")
        .attr("offset", "50%")
        .attr("stop-color", "gray");

    gradient.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", "#0571b0");

    legend2.append('text')
        .attr('x',0)
        .attr('y',80)
        .style('text-anchor', 'start')
        .html('Topic is present in:');

    legend2.append("rect")
        .attr("width", 300)
        .attr("height", 20)
        .attr('x',0)
        .attr('y',100)
        .style("fill", "url(#gradient)");

    legend2.append('text')
        .attr('x',0)
        .attr('y',140)
        .style('text-anchor', 'start')
        .html('AskWomen');
    legend2.append('text')
        .attr('x',150)
        .attr('y',140)
        .style('text-anchor', 'center')
        .html('both');
    legend2.append('text')
        .attr('x',300)
        .attr('y',140)
        .style('text-anchor', 'end')
        .html('AskMen');

    legend2.append('text')
        .attr('x',10)
        .attr('y',250)
        .style('text-anchor', 'start')
        .html('- size: overall occurence of topic');
    
    legend2.append('text')
        .attr('x',10)
        .attr('y',300)
        .style('text-anchor', 'start')
        .html('- position: 2-dim topic embedding');

});

