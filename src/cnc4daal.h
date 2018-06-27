/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef _CNC4DAAL_INCLUDED_
#define _CNC4DAAL_INCLUDED_

#include <array>
#include "daal.h"

template< typename T >
inline std::ostream & cnc_format( std::ostream& os, const daal::services::SharedPtr< T > & x )
{
    os << "SharedPtr<" << x.get() << ">";
    return os;
}

#ifdef _DIST_
# include <cnc/dist_cnc.h>
#else
# include <cnc/cnc.h>
#endif
#include <cnc/debug.h>

namespace CnC {

// We define a DataArchive that uses CnC's serializer directly.
// The basic trick is in providing write and read methods.
// For a final solution a narrower interface might be possible
// For example the different get/copyArchive methods should be
// on a separate level.
class DataArchive : public daal::data_management::interface1::DataArchiveImpl
{
public:
    /**
     *  Constructor of an empty data archive
     */
    DataArchive(CnC::serializer & s)
        : DataArchiveImpl(),
          ser(s)
    {
    }

    void write(daal::byte *ptr, size_t size) DAAL_C11_OVERRIDE
    {
        assert(ser.is_packing());
        ser & CnC::chunk< daal::byte, CnC::no_alloc >(ptr, size);
    }

    void read(daal::byte *ptr, size_t size) DAAL_C11_OVERRIDE
    {
        assert(ser.is_unpacking());
        ser & CnC::chunk< daal::byte, CnC::no_alloc >(ptr, size);
    }

    size_t getSizeOfArchive() const DAAL_C11_OVERRIDE
    {
        return ser.get_body_size();
    }

    daal::byte *getArchiveAsArray() DAAL_C11_OVERRIDE
    {
        return (daal::byte*)ser.get_body();
    }

    // we do not support this for now, we do not want to encourage use of
    // methods that require copying. If needed, it should live on a higher level IF.
    size_t copyArchiveToArray( daal::byte *ptr, size_t maxLength ) const DAAL_C11_OVERRIDE
    {
        assert(false);
        return 0;
    }

    // we do not support this for now, we do not want to encourage use of
    // methods that require copying. If needed, it should live on a higher level IF.
    services::SharedPtr<daal::byte> getArchiveAsArraySharedPtr() const DAAL_C11_OVERRIDE
    {
        assert(false);
        return services::SharedPtr<daal::byte>();
    }

    // we do not support this for now, we do not want to encourage use of
    // methods that require copying. If needed, it should live on a higher level IF.
    std::string getArchiveAsString() DAAL_C11_OVERRIDE
    {
        assert(false);
        return std::string();
    }

private:
    // we might want this to be a shared pointer
    CnC::serializer & ser;
};

// to be used to make a data-collection be consumed but not serve as controler
class no_mapper {};

// A data(collection) controlled step-collection, is its own tag-collection
template< typename DTag, typename CTag, typename D2CMap, typename UserStep, typename Tuner >
class dc_step_collection : public step_collection< UserStep, Tuner >, public tag_collection< CTag, Tuner >
{
public:
    template< typename Context >
    dc_step_collection( Context & ctxt, const std::string & name = std::string())
        : step_collection< UserStep, Tuner >(ctxt, name),
          tag_collection< CTag, Tuner >(ctxt, name),
          m_mapper()
    {
        tag_collection< CTag, Tuner >::prescribes(*this, ctxt);
    }
    
    template< typename Context >
    dc_step_collection( Context & ctxt, const Tuner & tnr, const std::string & name = std::string())
        : step_collection< UserStep, Tuner >(ctxt, tnr, name),
          tag_collection< CTag, Tuner >(ctxt, name, tnr),
          m_mapper()
    {
        tag_collection< CTag, Tuner >::prescribes(*this, ctxt);
    }
    
    template< typename Context, typename Arg >
    dc_step_collection( Context & ctxt, const Tuner & tnr, const std::string & name, Arg & arg)
        : step_collection< UserStep, Tuner >(ctxt, tnr, name),
          tag_collection< CTag, Tuner >(ctxt, name, tnr),
          m_mapper()
    {
        tag_collection< CTag, Tuner >::prescribes(*this, arg);
    }

    template< typename Context, typename Arg >
    dc_step_collection(Context & ctxt, const std::string & name, Arg & arg)
        : step_collection< UserStep, Tuner >(ctxt, name),
          tag_collection< CTag, Tuner >(ctxt, name),
          m_mapper()
    {
        tag_collection< CTag, Tuner >::prescribes(*this, arg);
    }


    // Declare this step-collection as consumer of given item-collection plus a controllee of it's tag
    // We allow a tranformation of the controlling data tag to the control tag
    template< typename Item, typename ITuner >
    void consume_as_controllee(CnC::item_collection< DTag, Item, ITuner > & dcoll)
    {
        step_collection< UserStep, Tuner >::consumes(dcoll);
        struct callback_type : public CnC::item_collection< DTag, Item, ITuner >::callback_type
        {
            callback_type(dc_step_collection< DTag, CTag, D2CMap, UserStep, Tuner > & tc) : m_dctagcoll(tc) {}

            virtual void on_put( const DTag & tag, const Item & )
            {
                m_dctagcoll.on_put(tag);
            }

            dc_step_collection< DTag, CTag, D2CMap, UserStep, Tuner > & m_dctagcoll;
        };
        callback_type * cb = new callback_type(*this); // FIXME: memory leak
        dcoll.on_put(cb);
    }

    using step_collection< UserStep, Tuner >::consumes;

    template< typename Item, typename ITuner >
    void consumes( CnC::item_collection< DTag, Item, ITuner > & dcoll, const D2CMap & mapper )
    {
        m_mapper = mapper;
        consume_as_controllee(dcoll);
    }
    
    template< typename Item, typename ITuner >
    void consumes(CnC::item_collection< DTag, Item, ITuner > & dcoll, const no_mapper)
    {
        step_collection< UserStep, Tuner >::consumes(dcoll);
    }

    void on_put_(const CTag & tag)
    {
        tag_collection< CTag, Tuner >::put(tag);
    }
    void on_put_(const std::vector<CTag> & tags)
    {
        for(auto i = tags.begin(); i != tags.end(); ++i ) {
            tag_collection< CTag, Tuner >::put(*i);
        }
    }
    void on_put(const DTag & tag)
    {
        on_put_(m_mapper(tag));
    }

private:
    D2CMap  m_mapper;
};

template< typename T >
struct identityMap {
    T operator()(const T & tag) {return tag;}
};

template< typename T, int tag >
struct singletonMap {
    int operator()(const T &) {return tag;}
};

template< typename A, typename T, int tag >
struct dimMap {
    T operator()(const A & a) {return std::get<tag>(a);}
};


#ifdef _DIST_

    inline void serialize_si( CnC::serializer & ser, daal::services::SharedPtr< daal::data_management::SerializationIface > & ptr )
    {
        if( ser.is_packing() ) {
            bool is = ptr ? true : false;
            ser & is;
            if(ptr) {
                // Create a data archive to serialize the numeric table
                // we use our own DataArchive to avoid double/triple copying
                // by directly serializing into our serializer
                data_management::InputDataArchive dataArch(new CnC::DataArchive(ser));
                /* Serialize the numeric table into the data archive */
                ptr->serialize(dataArch);
            }
        } else if( ser.is_unpacking() ) {
            bool is;
            ser & is;
            if(is) {
                // Create a data archive to deserialize the numeric table
                // we use our own DataArchive to avoid double/triple copying
                // by directly deserializing from our serializer
                daal::data_management::OutputDataArchive dataArch(new CnC::DataArchive(ser));
                /* Deserialize the numeric table from the data archive */
                ptr = dataArch.getAsSharedPtr();
            } else {
                ptr.reset();
            }
        }
    }

    template< typename T >
    void serialize(CnC::serializer & ser, daal::services::SharedPtr< T > & ptr)
    {
        serialize_si(ser, reinterpret_cast< daal::services::SharedPtr< daal::data_management::SerializationIface > & >(ptr));
    }

    static inline void serialize( serializer & ser, daal::data_management::NumericTablePtr *& t ) {
        ser & chunk< daal::data_management::NumericTablePtr >(t, 1);
    }

#endif //_DIST_
} // namespace CnC

#endif // _CNC4DAAL_INCLUDED_
