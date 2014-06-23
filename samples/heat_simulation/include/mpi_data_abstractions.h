// Created with help from Chuck Knight's Blog
// http://chuckaknight.wordpress.com/2013/03/13/intrinsic-type-conversion-using-template-specialization/

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

#include "sys/types.h"

namespace MPI_Data_Abstraction
{
typedef enum
{
    type_unknown = 0,
    type_bool,
    type_char,
    type_unsigned_char,
    type_short,
    type_unsigned_short,
    type_int,
    type_unsigned_int,
    type_long,
    type_unsigned_long,
    type_float,
    type_double,
} Data_Type;
}

template <class T>
MPI_Data_Abstraction::Data_Type get_abstraction_data_type()
{
    throw std::runtime_error("Data type not supported");
}

template<>
inline MPI_Data_Abstraction::Data_Type get_abstraction_data_type<bool>()
{
    return MPI_Data_Abstraction::type_bool;
}

template<>
inline MPI_Data_Abstraction::Data_Type get_abstraction_data_type<char>()
{
    return MPI_Data_Abstraction::type_char;
}

template<>
inline MPI_Data_Abstraction::Data_Type get_abstraction_data_type<unsigned char>()
{
    return MPI_Data_Abstraction::type_unsigned_char;
}

template<>
inline MPI_Data_Abstraction::Data_Type get_abstraction_data_type<short>()
{
    return MPI_Data_Abstraction::type_short;
}

template<>
inline MPI_Data_Abstraction::Data_Type get_abstraction_data_type<unsigned short>()
{
    return MPI_Data_Abstraction::type_unsigned_short;
}

template<>
inline MPI_Data_Abstraction::Data_Type get_abstraction_data_type<int>()
{
    return MPI_Data_Abstraction::type_int;
}

template<>
inline MPI_Data_Abstraction::Data_Type get_abstraction_data_type<unsigned int>()
{
    return MPI_Data_Abstraction::type_unsigned_int;
}

template<>
inline MPI_Data_Abstraction::Data_Type get_abstraction_data_type<long>()
{
    return MPI_Data_Abstraction::type_long;
}

template<>
inline MPI_Data_Abstraction::Data_Type get_abstraction_data_type<unsigned long>()
{
    return MPI_Data_Abstraction::type_unsigned_long;
}

template<>
inline MPI_Data_Abstraction::Data_Type get_abstraction_data_type<float>()
{
    return MPI_Data_Abstraction::type_float;
}

template<>
inline MPI_Data_Abstraction::Data_Type get_abstraction_data_type<double>()
{
    return MPI_Data_Abstraction::type_double;
}

