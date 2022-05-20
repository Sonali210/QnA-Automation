$(document).ready(function(){
    $(".open-close-button").click(function(){

        if($(".open-close-button").hasClass("stat-close")){

            $("#myForm").css('opacity', 0)
            .slideDown('slow')
            .animate(
              { opacity: 1 },
              { queue: false, duration: 'slow' }
            );
            $(".open-close-button").removeClass('fa-comment');
            $(".open-close-button").addClass("fa-times-thin");
            $(".open-close-button").removeClass('stat-close');
            $(".open-close-button").addClass("stat-open");

        } else {

            $("#myForm").slideUp("slow");
            $(".open-close-button").removeClass('fa-times-thin');
            $(".open-close-button").addClass("fa-comment");
            $(".open-close-button").removeClass('stat-open');
            $(".open-close-button").addClass("stat-close");
            
        }
      
    });
    $(".close-button").click(function(){
      $("#myForm").slideUp("slow");
      $(".open-close-button").removeClass('fa-times-thin');
      $(".open-close-button").addClass("fa-comment");
      $(".open-close-button").removeClass('stat-open');
      $(".open-close-button").addClass("stat-close");
    });
  });


var me = {};
me.avatar = "https://lh6.googleusercontent.com/-lr2nyjhhjXw/AAAAAAAAAAI/AAAAAAAARmE/MdtfUmC0M4s/photo.jpg?sz=48";

var you = {};
you.avatar = "https://a11.t26.net/taringa/avatares/9/1/2/F/7/8/Demon_King1/48x48_5C5.jpg";

function formatAMPM(date) {
    var hours = date.getHours();
    var minutes = date.getMinutes();
    var ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12;
    hours = hours ? hours : 12; // the hour '0' should be '12'
    minutes = minutes < 10 ? '0'+minutes : minutes;
    var strTime = hours + ':' + minutes + ' ' + ampm;
    return strTime;
}            

//-- No use time. It is a javaScript effect.
function insertChat(who, text, time){
    if (time === undefined){
        time = 0;
    }
    var control = "";
    var date = formatAMPM(new Date());
    
    if (who == "you"){
        control = '<li style="width:100%">' +
                        '<div class="msj macro">' +
                        '<div class="avatar"><img class="img-circle" style="width:100%;border-radius: 50%;" src="static/swar_ai/assets/img/chatbot_dp.png" /></div>' +
                            '<div class="text text-l">' +
                                '<p>'+ text +'</p>' +
                                '<p><small>'+date+'</small></p>' +
                            '</div>' +
                        '</div>' +
                    '</li>';                    
    }else{
        control = '<li style="width:100%;">' +
                        '<div class="msj-rta macro">' +
                            '<div class="text text-r">' +
                                '<p>'+text+'</p>' +
                                '<p><small>'+date+'</small></p>' +
                            '</div>' +
                        '<div class="avatar" style="padding:0px 0px 0px 10px !important;"><img class="img-circle" style="width:100%;border-radius: 50%" src="static/swar_ai/assets/img/user_dp.jpg" /></div>' +                                
                  '</li>';
    }
    setTimeout(
        function(){                        
            $(".messages").append(control).scrollTop($(".messages").prop('scrollHeight'));
        }, time);
    
}

function resetChat(){
    $(".messages").empty();
}

function getBotResponse(text) {
    var rawText = text;
    $.get("/get", { msg: rawText }).done(function(data) {
        insertChat("you", data); ;
    });
  }

$(".mytext").on("keydown", function(e){
    if (e.which == 13){
        var text = $(this).val();
        if (text !== ""){
            insertChat("me", text);              
            $(this).val('');
        }
        getBotResponse(text);
    }
});

$('body > div > div > div:nth-child(2) > span').click(function(){
    $(".mytext").trigger({type: 'keydown', which: 13, keyCode: 13});
})

//-- Clear Chat
resetChat();

//-- Print Messages
// insertChat("me", "Hello Tom...", 0);  
insertChat("you", "Hi!!", 1500);
// insertChat("me", "What would you like to talk about today?", 3500);
// insertChat("you", "Tell me a joke",7000);
// insertChat("me", "Spaceman: Computer! Computer! Do we bring battery?!", 9500);
// insertChat("you", "LOL", 12000);


//-- NOTE: No use time on insertChat.