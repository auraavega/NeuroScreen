let eegChart = null;
let waveChart = null;



async function analyze(){



    const file =
    document.getElementById("file").files[0];



    const name =
    document.getElementById("username").value;



    // =====================
    // VALIDASI INPUT
    // =====================


    if(!name){

        alert(
            "Masukkan nama pengguna terlebih dahulu"
        );

        return;

    }


    if(!file){

        alert(
            "Silakan upload file EEG (.txt)"
        );

        return;

    }






    // =====================
    // LOADING
    // =====================


    document
    .getElementById("loading")
    .classList
    .remove("hidden");






    const button =
    document.querySelector("button");


    button.innerHTML =
    "⏳ Processing EEG...";


    button.disabled = true;






    // =====================
    // SEND DATA
    // =====================


    let form =
    new FormData();


    form.append(
        "file",
        file
    );


    form.append(
        "username",
        name
    );







    try{


        let response =
        await fetch(
            "/predict",
            {

                method:"POST",

                body:form

            }
        );



        let data =
        await response.json();







        // =====================
        // SHOW RESULT
        // =====================



        document
        .getElementById("result")
        .classList
        .remove("hidden");



        document
        .getElementById("hello")
        .innerHTML =

        "Hello "
        +
        data.name
        +
        " 👋";




        document
        .getElementById("score")
        .innerHTML =

        data.risk_score;








        // =====================
        // EEG SIGNAL GRAPH
        // =====================


        let ctx =
        document
        .getElementById(
            "eegChart"
        );



        if(eegChart){

            eegChart.destroy();

        }





        eegChart =
        new Chart(

            ctx,

            {

            type:"line",


            data:{


                labels:

                data.eeg_signal.channel1.map(

                    (x,i)=>i

                ),



                datasets:[


                {


                label:
                "EEG Channel 1",


                data:
                data.eeg_signal.channel1,


                borderWidth:1,


                pointRadius:0


                },



                {


                label:
                "EEG Channel 2",


                data:
                data.eeg_signal.channel2,


                borderWidth:1,


                pointRadius:0


                }


                ]

            },



            options:{


                responsive:true,


                animation:false,


                scales:{


                    x:{


                        title:{


                            display:true,


                            text:"Sample"

                        }


                    },


                    y:{


                        title:{


                            display:true,


                            text:"Amplitude"

                        }


                    }


                }



            }



            }

        );









        // =====================
        // BRAIN WAVE GRAPH
        // =====================


        let ctx2 =

        document
        .getElementById(
            "waveChart"
        );




        if(waveChart){

            waveChart.destroy();

        }





        waveChart =

        new Chart(

            ctx2,

            {


            type:"bar",



            data:{



                labels:

                Object.keys(

                    data.brainwave

                ),




                datasets:[


                {


                label:

                "Brain Wave Power",



                data:

                Object.values(

                    data.brainwave

                )



                }



                ]

            },



            options:{


                responsive:true,


                plugins:{


                    legend:{


                        display:true

                    }


                }


            }



            }

        );







    }



    catch(error){


        console.error(error);


        alert(

            "Terjadi error saat analisis EEG"

        );


    }







    finally{


        document
        .getElementById("loading")
        .classList
        .add("hidden");



        button.innerHTML =

        "Analisis EEG";



        button.disabled = false;



    }



}