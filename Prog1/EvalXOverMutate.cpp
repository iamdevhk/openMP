#include <iostream> // cout
#include <stdlib.h> // rand
#include <math.h>   // sqrt, pow
#include <omp.h>    // OpenMP
#include <string.h> // memset
#include "Timer.h"
#include "Trip.h"
#include <algorithm>

#define CHROMOSOMES 50000 // 50000 different trips
#define CITIES 36         // 36 cities = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
#define TOP_X 25000       // top optimal 25%
#define MUTATE_RATE 50    // optimal 50%

using namespace std;

// find city index based on char passed -- in used in multiple functions
int findCityIndex(char city)
{
    return city >= 'A' ? city - 'A' : city - '0' + 26;
}

/*
 * Evaluates the distance/fitness of each trip/chromosome and sorts them out in ascending order.
 */
void evaluate(Trip trip[CHROMOSOMES], int coordinates[CITIES][2])
{

// parallelizing using openmp
#pragma omp parallel for
    for (int chromosomes = 0; chromosomes < CHROMOSOMES; chromosomes++)
    {
        double x1 = 0, y1 = 0, distance = 0;
        double x2, y2;
        int index = -1;
        string chromosomeStr = trip[chromosomes].itinerary;

        // calculates total distance for one entire trip
        for (int itr = 0; itr < chromosomeStr.length(); itr++)
        {
            index = findCityIndex(chromosomeStr[itr]);
            x2 = coordinates[index][0];
            y2 = coordinates[index][1];
            distance = distance + (sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2)));
            x1 = x2;
            y1 = y2;
        }
        // assign total distance
        trip[chromosomes].fitness = distance;
    }

    // sorts based on fitness
    std::sort(trip, trip + CHROMOSOMES, [](Trip trip1, Trip trip2)
              { return trip1.fitness < trip2.fitness; });
}

//function to find distance between two coordinates
double findDistance(int x1, int y1, int x2, int y2)
{
    return std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
}

//function to find complement offspring2 based on input offspring1
std::string findComplement(const std::string &input)
{
    const char original[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    const char complement[] = "9876543210ZYXWVUTSRQPONMLKJIHGFEDCBA";

    std::string complementString(input);

    // Create a character mapping table for faster lookups
    char charMap[256];
    for (int i = 0; i < CITIES; i++)
    {
        charMap[original[i]] = complement[i];
    }

   
    for (int i = 0; i < CITIES; i++)
    {
        char c = input[i];

        // Use the character mapping table for direct lookup
        complementString[i] = charMap[c];
    }

    return complementString;
}

/*
 * Generates new TOP_X offsprings from TOP_X parents.
 * The i and i+1 offsprings are created from the i and i+1  parents
 */
void crossover(Trip parents[TOP_X], Trip offsprings[TOP_X], int coordinates[CITIES][2])
{

    //iterate thorugh top 50%
#pragma omp parallel for
    for (int offspring = 0; offspring < TOP_X; offspring += 2)
    {

        string parent1 = parents[offspring].itinerary;
        string parent2 = parents[offspring + 1].itinerary;
        string child1;
        int visited[CITIES] = {0};
        char offSpring1[CITIES];
        char offSpring2[CITIES];
        int parent1Index = 1;
        int parent2Index = 0;
        int currentIndex = 1;


        char firstParent1Char = parent1.at(0);
        child1.push_back(firstParent1Char);
        char previousCityIndex = findCityIndex(firstParent1Char);
        visited[previousCityIndex] = 1;

        //iterate all 36 chars in the itinerary
        while (currentIndex < CITIES)
        {
            char currentCity1 = parent1.at(parent1Index);
            char currentCity2 = parent2.at(parent2Index);
            int curCity1Index = findCityIndex(currentCity1);
            int curCity2Index = findCityIndex(currentCity2);

            //if both the indexes are not visited calculate distances of both from current city
            if (!visited[curCity1Index] && !visited[curCity2Index])
            {
                int x1 = coordinates[curCity1Index][0];
                int y1 = coordinates[curCity1Index][1];
                int x2 = coordinates[previousCityIndex][0];
                int y2 = coordinates[previousCityIndex][1];
                double curCity1Distance = 0.0;
                double curCity2Distance = 0.0;

                curCity1Distance = findDistance(x1, y1, x2, y2);
                x1 = coordinates[curCity2Index][0];
                y1 = coordinates[curCity2Index][1];
                curCity2Distance = findDistance(x1, y1, x2, y2);

                //if city1 distance is small
                if (curCity1Distance < curCity2Distance)
                {
                    //the next city in the itinerary would be city1
                    child1 = child1 + currentCity1;
                    previousCityIndex = curCity1Index;
                    visited[curCity1Index] = 1;
                    parent1Index++;
                }
                else
                {
                    //else the next city would be city2
                    child1 = child1 + currentCity2;
                    previousCityIndex = curCity2Index;
                    visited[curCity2Index] = 1;
                    parent2Index++;
                }
            }
            //if city1 is alreadu visited and city2 is not visited
            else if (visited[curCity1Index] && !visited[curCity2Index])
            {
                //the next city in the itinerary would be city2
                child1 = child1 + currentCity2;
                previousCityIndex = curCity2Index;
                visited[curCity2Index] = 1;
                parent1Index++;
                parent2Index++;
                
            }
            //if city1 is not visited and city 2 is already visited
            else if (!visited[curCity1Index] && visited[curCity2Index])
            {
                //the next city in the itinerary would be city1
                child1 = child1 + currentCity1;
                previousCityIndex = curCity1Index;
                visited[curCity1Index] = 1;
                parent1Index++;
                parent2Index++;
            }
            else
            {
                currentIndex--;
                parent1Index++;
                parent2Index++;    
            }
            currentIndex++;
        }

        
        //add child 1 value to offSpring1
        strcpy(offSpring1, child1.c_str());

        //find the complement of offspring1 and store in offspring2
        strncpy(offSpring2, findComplement(offSpring1).c_str(), CITIES);
        offSpring2[CITIES] = '\0';

        //store it to the itinerary
        strcpy(offsprings[offspring].itinerary, offSpring1);
        strcpy(offsprings[offspring + 1].itinerary, offSpring2);
    }
}

/*
 * Mutate - swap a pair of genes in each offspring.
 */
void mutate(Trip offsprings[TOP_X])
{

    //loop through top 50%
    for (int itr = 0; itr < TOP_X; itr++)
    {
        char offSpringTmp[CITIES];

        //copy itinerary to  a temporary string
        strcpy(offSpringTmp, offsprings[itr].itinerary);
        string offSpringTmpStr = offSpringTmp;
        int randomNum = rand() % 100;
        if (randomNum <= MUTATE_RATE)
        {
            int mutationPoint1 = rand() % CITIES;
            int mutationPoint2 = rand() % CITIES;
            swap(offSpringTmpStr[mutationPoint1], offSpringTmpStr[mutationPoint2]);
            strcpy(offsprings[itr].itinerary, offSpringTmpStr.c_str());
        }
    }
}