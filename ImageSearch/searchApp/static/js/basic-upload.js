$(function () {
  $(".js-upload-photos").click(function () {
    $("#fileupload").click();
  });

  $("#fileupload").fileupload({
    dataType: 'json',
    done: function (e, data) {
      console.log(data.result.result)
         console.log("succeess");
        $("#photo_file_url").attr("src",data.result.photo_file_url)
         $("#results").empty();
         for (var i=0;i<data.result.result.length;++i){
            var figure=document.createElement("figure");
            var image=document.createElement("IMG");
            image.src="/static/images/"+data.result.result[i];
            var figcaption=document.createElement("figcaption");
            figcaption.append("/static/images/"+data.result.result[i]);
            figure.appendChild(image);
            figure.appendChild(figcaption);
            $("#results").append(figure);
         }
      }
  });

});
