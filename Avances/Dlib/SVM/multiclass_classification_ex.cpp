// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

	Esto es un ejemplo ilustrativo del uso de un clasificador multiclase de la
	libreria Dlib. Especificamente en este ejemplo clasificaremos puntos en las 
	siguiente clases:

	-	Clase 1: Puntos muy cerca del origen.
	- 	Clase 2: Puntos en un radio de 10 de unidades alrededor del circulo.
	- 	Clase 3: Puntos que estan en un circulo de radio 4 pero no alrededor del origen.


    This is an example illustrating the use of the multiclass classification tools  
    from the dlib C++ Library.  Specifically, this example will make points from 
    three classes and show you how to train a multiclass classifier to recognize 
    these three classes.

    The classes are as follows:
        - class 1: points very close to the origin
        - class 2: points on the circle of radius 10 around the origin
        - class 3: points that are on a circle of radius 4 but not around the origin at all
*/

#include <dlib/svm_threaded.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include <dlib/rand.h>

using namespace std;
using namespace dlib;

// Our data will be 2-dimensional data. So declare an appropriate type to contain these points.

typedef matrix<double,2,1> sample_type;

// ----------------------------------------------------------------------------------------

void generate_data (
    std::vector<sample_type>& samples,
    std::vector<double>& labels
);
/*!
    ensures
        - make some 3 class data as described above.  
        - Create 60 points from class 1
        - Create 70 points from class 2
        - Create 80 points from class 3
!*/

// ----------------------------------------------------------------------------------------

int main()
{
    try
    {
        std::vector<sample_type> samples;
        std::vector<double> labels;

        // First, get our labeled set of training data
        generate_data(samples, labels);

        cout << "samples.size(): "<< samples.size() << endl;
		
		/*
			Mostramos la muestra (caracteristicas) y la etiqueta
		*/
		for(int i = 0; i< samples.size(); i++)
				cout<<"("<<setw(1)<<samples[i](0,0)<<" , "<<samples[i](1,0)
				<<setw(1)<<")"<<setw(10)<<"--->"<<setw(10)<<" Etiqueta: "<<labels[i]<<endl;
		
        // The main object in this example program is the one_vs_one_trainer.  It is essentially 
        // a container class for regular binary classifier trainer objects.  In particular, it 
        // uses the any_trainer object to store any kind of trainer object that implements a 
        // .train(samples,labels) function which returns some kind of learned decision function.  
        // It uses these binary classifiers to construct a voting multiclass classifier.  If 
        // there are N classes then it trains N*(N-1)/2 binary classifiers, one for each pair of 
        // labels, which then vote on the label of a sample.
        //
        // In this example program we will work with a one_vs_one_trainer object which stores any 
        // kind of trainer that uses our sample_type samples.

		/*
			El tipo de datos one_vs_one_trainer, es un contenedor de clasificadores
			binarios, este contiene un parametro de tipo llamado any_trainer
			que a su vez tiene un parametro de tipo que es el tipo de dato de 
			la muestra.

			Esto nos permite almacenar en este contendor cualquier clasificador
			binario que trabaje que tenga como muestra el tipo de datos
			sample_type

			Si existen 3 clases, entonces el contenedor entrena N*(N-1)/2 clasificadores
			binarios. 1 entrenamiento por cada par de clases o etiquetas.
		*/
        typedef one_vs_one_trainer<any_trainer<sample_type> > ovo_trainer;


        // Instanciamos nuestro contenedor
        ovo_trainer trainer;


        // Next, we will make two different binary classification trainer objects.  One
        // which uses kernel ridge regression and RBF kernels and another which uses a
        // support vector machine and polynomial kernels.  The particular details don't matter.
        // The point of this part of the example is that you can use any kind of trainer object
        // with the one_vs_one_trainer.
        typedef polynomial_kernel<sample_type> poly_kernel;
       //typedef radial_basis_kernel<sample_type> rbf_kernel;

        // make the binary trainers and set some parameters
        //krr_trainer<rbf_kernel> rbf_trainer;
        svm_nu_trainer<poly_kernel> poly_trainer;
        poly_trainer.set_kernel(poly_kernel(0.1, 1, 2));
        //rbf_trainer.set_kernel(rbf_kernel(0.1));


        // Now tell the one_vs_one_trainer that, by default, it should use the rbf_trainer
        // to solve the individual binary classification subproblems.
        //trainer.set_trainer(rbf_trainer);
        // We can also get more specific.  Here we tell the one_vs_one_trainer to use the
        // poly_trainer to solve the class 1 vs class 2 subproblem.  All the others will
        // still be solved with the rbf_trainer.
		trainer.set_trainer(poly_trainer);
        //trainer.set_trainer(poly_trainer, 1, 2);

        
		// Now let's do 5-fold cross-validation using the one_vs_one_trainer we just setup.
        // As an aside, always shuffle the order of the samples before doing cross validation.  
        // For a discussion of why this is a good idea see the svm_ex.cpp example.
        randomize_samples(samples, labels);
        cout << "cross validation: \n" << cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;
        // The output is shown below.  It is the confusion matrix which describes the results.  Each row 
        // corresponds to a class of data and each column to a prediction.  Reading from top to bottom, 
        // the rows correspond to the class labels if the labels have been listed in sorted order.  So the
        // top row corresponds to class 1, the middle row to class 2, and the bottom row to class 3.  The
        // columns are organized similarly, with the left most column showing how many samples were predicted
        // as members of class 1.
        // 
        // So in the results below we can see that, for the class 1 samples, 60 of them were correctly predicted
        // to be members of class 1 and 0 were incorrectly classified.  Similarly, the other two classes of data
        // are perfectly classified.
        /*
            cross validation: 
            60  0  0 
            0 70  0 
            0  0 80 
        */

        // Next, if you wanted to obtain the decision rule learned by a one_vs_one_trainer you 
        // would store it into a one_vs_one_decision_function.
        one_vs_one_decision_function<ovo_trainer> df = trainer.train(samples, labels);

        cout <<"Samples: ("<<samples[0](0,0)<<" , "<<samples[0](1,0)<<") prediccion: "<<
		df(samples[0])  << ", valor real: "<< labels[0] << endl;
        
		cout <<"Samples: ("<<samples[90](0,0)<<" , "<<samples[90](1,0)<<") prediccion: "<<
		df(samples[90])  << ", valor real: "<< labels[90] << endl;

		cout <<"Samples: ("<<samples[2](0,0)<<" , "<<samples[2](1,0)<<") prediccion: "<<
		df(samples[2])  << ", valor real: "<< labels[2] << endl;
        
		// The output is:
        /*
            predicted label: 2, true label: 2
            predicted label: 1, true label: 1
        */


        // If you want to save a one_vs_one_decision_function to disk, you can do
        // so.  However, you must declare what kind of decision functions it contains. 
        one_vs_one_decision_function<ovo_trainer, 
        decision_function<poly_kernel>  // This is the output of the poly_trainer
      //  decision_function<rbf_kernel>    // This is the output of the rbf_trainer
        > df2, df3;


        // Put df into df2 and then save df2 to disk.  Note that we could have also said
        // df2 = trainer.train(samples, labels);  But doing it this way avoids retraining.
		cout<<"Serializamos las funciones de decision "<<endl;
        df2 = df;
        serialize("df.dat") << df2;
		
		cout<<"Deserializamos las funciones de decision y probamos"<<endl;
        // load the function back in from disk and store it in df3.  
        deserialize("df.dat") >> df3;


        // Test df3 to see that this worked.
        cout << endl;
        cout <<"Samples: ("<<samples[0](0,0)<<" , "<<samples[0](1,0)<<") prediccion: "<<
		df3(samples[0])  << ", valor real: "<< labels[0] << endl;
        
		cout <<"Samples: ("<<samples[90](0,0)<<" , "<<samples[90](1,0)<<") prediccion: "<<
		df3(samples[90])  << ", valor real: "<< labels[90] << endl;

		cout <<"Samples: ("<<samples[2](0,0)<<" , "<<samples[2](1,0)<<") prediccion: "<<
		df3(samples[2])  << ", valor real: "<< labels[2] << endl;
        
		cout << "test deserialized function: \n" << test_multiclass_decision_function(df3, samples, labels) << endl;





        // Finally, if you want to get the binary classifiers from inside a multiclass decision
        // function you can do it by calling get_binary_decision_functions() like so:
        one_vs_one_decision_function<ovo_trainer>::binary_function_table functs;
        functs = df.get_binary_decision_functions();
        cout << "number of binary decision functions in df: " << functs.size() << endl;
     
    }
    catch (std::exception& e)
    {
        cout << "exception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

void generate_data (
    std::vector<sample_type>& samples,
    std::vector<double>& labels
)
{
    const long num = 50;

    sample_type m;

    dlib::rand rnd;


    // make some samples near the origin
    double radius = 0.5;
    for (long i = 0; i < num+10; ++i)
    {
        double sign = 1;
        if (rnd.get_random_double() < 0.5)
            sign = -1;
        m(0) = 2*radius*rnd.get_random_double()-radius;
        m(1) = sign*sqrt(radius*radius - m(0)*m(0));

        // add this sample to our set of training samples 
        samples.push_back(m);
        labels.push_back(1);
    }

    // make some samples in a circle around the origin but far away
    radius = 10.0;
    for (long i = 0; i < num+20; ++i)
    {
        double sign = 1;
        if (rnd.get_random_double() < 0.5)
            sign = -1;
        m(0) = 2*radius*rnd.get_random_double()-radius;
        m(1) = sign*sqrt(radius*radius - m(0)*m(0));

        // add this sample to our set of training samples 
        samples.push_back(m);
        labels.push_back(2);
    }

    // make some samples in a circle around the point (25,25) 
    radius = 4.0;
    for (long i = 0; i < num+30; ++i)
    {
        double sign = 1;
        if (rnd.get_random_double() < 0.5)
            sign = -1;
        m(0) = 2*radius*rnd.get_random_double()-radius;
        m(1) = sign*sqrt(radius*radius - m(0)*m(0));

        // translate this point away from the origin
        m(0) += 25;
        m(1) += 25;

        // add this sample to our set of training samples 
        samples.push_back(m);
        labels.push_back(3);
    }
}

// ----------------------------------------------------------------------------------------

